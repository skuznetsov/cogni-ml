require "spec"
require "../examples/support/hy_nomas_reference_refiner_trace"

describe HyNomasReferenceRefinerTrace do
  it "pins the source-comparable valid-prefix tensor order and shapes" do
    HyNomasReferenceRefinerTrace::TENSOR_ORDER.should eq([
      "input_compact_valid", "masked_mean", "timestep",
      "refiner_timestep_sinusoidal", "refiner_timestep_mlp1",
      "refiner_timestep_silu", "refiner_timestep", "context_linear1",
      "context_silu", "context_embed", "c", "input_embed",
      "time_in.sinusoidal", "time_in.mlp1", "time_in.silu", "vec",
    ])

    {"input_compact_valid"         => [1, 263, 3584],
     "masked_mean"                 => [1, 3584],
     "timestep"                    => [1],
     "refiner_timestep_sinusoidal" => [1, 256],
     "refiner_timestep_mlp1"       => [1, 2048],
     "refiner_timestep_silu"       => [1, 2048],
     "refiner_timestep"            => [1, 2048],
     "context_linear1"             => [1, 2048],
     "context_silu"                => [1, 2048],
     "context_embed"               => [1, 2048],
     "c"                           => [1, 2048],
     "input_embed"                 => [1, 263, 2048],
     "time_in.sinusoidal"          => [1, 256],
     "time_in.mlp1"                => [1, 2048],
     "time_in.silu"                => [1, 2048],
     "vec"                         => [1, 2048]}.each do |name, expected_shape|
      dtype, shape = HyNomasReferenceRefinerTrace.tensor_spec(name, 263)
      dtype.should eq("F32")
      shape.should eq(expected_shape)
    end

    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.tensor_spec("input_embed", 0)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.tensor_spec("unknown", 263)
    end
  end

  it "requires zero in-encode allocation/readback/waits and one terminal drain" do
    HyNomasReferenceRefinerTrace.validate_execution!(0_i64, 0_i64, 0, 1).should be_nil

    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.validate_execution!(1_i64, 0_i64, 0, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.validate_execution!(0_i64, 1_i64, 0, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.validate_execution!(0_i64, 0_i64, 1, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.validate_execution!(0_i64, 0_i64, 0, 2)
    end
  end

  it "marks prelude BF16 control as disabled by default and never full prelude parity" do
    disabled = HyNomasReferenceRefinerTrace.prelude_metadata
    disabled[:prelude_input_bf16_control].should be_false
    disabled[:full_bf16_prelude_parity].should be_false
    disabled[:prelude_input_bf16_control_scope].should eq("disabled")

    enabled = HyNomasReferenceRefinerTrace.prelude_metadata(true)
    enabled[:prelude_input_bf16_control].should be_true
    enabled[:full_bf16_prelude_parity].should be_false
    enabled[:prelude_input_bf16_control_scope].should contain("masked-mean sum")
  end

  it "fails closed before capture when refiner checkpoints are missing or unordered" do
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.tensor_sources({} of String => ML::MetalBuffer, 263)
    end

    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerTrace.write_trace!("relative.json",
        reference_latent_artifact_sha256: "a" * 64,
        timeline_manifest_sha256: "b" * 64,
        cache_fixture_manifest_sha256: "c" * 64,
        valid_text_tokens: 263,
        sources: [] of HyNomasReferenceRefinerTrace::TensorSource,
        encode_allocations: 0_i64,
        encode_readbacks: 0_i64,
        preterminal_waits: 0,
        terminal_waits: 1)
    end
  end

  it "requires fresh distinct sidecars before GPU execution" do
    root = File.tempname("hy-nomas-refiner-trace")
    Dir.mkdir(root)
    begin
      cache_manifest, cache_payload = HyNomasReferenceCacheProbe.checked_output_paths!(
        File.join(root, "cache.json"))
      trace_manifest, trace_payload = HyNomasReferenceRefinerTrace.checked_output_paths!(
        File.join(root, "trace.json"))
      HyNomasReferenceRefinerTrace.ensure_distinct_outputs!(cache_manifest, cache_payload,
        trace_manifest, trace_payload).should be_nil
      HyNomasReferenceRefinerTrace.ensure_fresh_outputs!(trace_manifest, trace_payload).should be_nil

      expect_raises(ArgumentError) do
        HyNomasReferenceRefinerTrace.ensure_distinct_outputs!(cache_manifest, cache_payload,
          cache_manifest, trace_payload)
      end

      File.write(trace_payload, "existing owner")
      expect_raises(ArgumentError) do
        HyNomasReferenceRefinerTrace.ensure_fresh_outputs!(trace_manifest, trace_payload)
      end
      File.read(trace_payload).should eq("existing owner")
    ensure
      Dir.glob(File.join(root, "*")).each { |path| File.delete(path) }
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "records source pins for the exact probe and refiner implementation" do
    pins = HyNomasReferenceRefinerTrace.compiled_source_sha256s
    pins.keys.should eq(HyNomasReferenceRefinerTrace::SOURCE_FILES)
    pins.values.each do |sha|
      HyNomasReferenceRefinerTrace.valid_sha256?(sha).should be_true
    end
  end
end
