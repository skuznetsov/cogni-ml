require "spec"
require "../examples/support/hy_nomas_reference_cache_probe"
require "../examples/support/hy_nomas_reference_refiner_trace"
require "../examples/support/hy_nomas_reference_refiner_blocks_trace"

describe HyNomasReferenceRefinerBlocksTrace do
  it "pins the full padded checkpoint inventory and source shapes" do
    HyNomasReferenceRefinerBlocksTrace::TENSOR_ORDER.should eq([
      "block0.input", "c",
      "block0.condition_silu", "block0.adaLN", "block0.gate_msa", "block0.gate_mlp",
      "block0.norm1", "block0.qkv", "block0.q", "block0.k", "block0.v",
      "block0.attention", "block0.proj", "block0.residual1", "block0.norm2",
      "block0.mlp_fc1", "block0.mlp_silu", "block0.mlp_fc2", "block0.output",
      "block1.input",
      "block1.condition_silu", "block1.adaLN", "block1.gate_msa", "block1.gate_mlp",
      "block1.norm1", "block1.qkv", "block1.q", "block1.k", "block1.v",
      "block1.attention", "block1.proj", "block1.residual1", "block1.norm2",
      "block1.mlp_fc1", "block1.mlp_silu", "block1.mlp_fc2", "block1.output",
    ])

    expected_shapes = {
      "block0.input"          => [1, 1280, 2048],
      "c"                     => [1, 2048],
      "block0.condition_silu" => [1, 2048],
      "block0.adaLN"          => [1, 4096],
      "block0.gate_msa"       => [1, 2048],
      "block0.gate_mlp"       => [1, 2048],
      "block0.norm1"          => [1, 1280, 2048],
      "block0.qkv"            => [1, 1280, 3, 2048],
      "block0.q"              => [1, 1280, 2048],
      "block0.k"              => [1, 1280, 2048],
      "block0.v"              => [1, 1280, 2048],
      "block0.attention"      => [1, 1280, 2048],
      "block0.proj"           => [1, 1280, 2048],
      "block0.residual1"      => [1, 1280, 2048],
      "block0.norm2"          => [1, 1280, 2048],
      "block0.mlp_fc1"        => [1, 1280, 8192],
      "block0.mlp_silu"       => [1, 1280, 8192],
      "block0.mlp_fc2"        => [1, 1280, 2048],
      "block0.output"         => [1, 1280, 2048],
      "block1.input"          => [1, 1280, 2048],
      "block1.condition_silu" => [1, 2048],
      "block1.adaLN"          => [1, 4096],
      "block1.gate_msa"       => [1, 2048],
      "block1.gate_mlp"       => [1, 2048],
      "block1.norm1"          => [1, 1280, 2048],
      "block1.qkv"            => [1, 1280, 3, 2048],
      "block1.q"              => [1, 1280, 2048],
      "block1.k"              => [1, 1280, 2048],
      "block1.v"              => [1, 1280, 2048],
      "block1.attention"      => [1, 1280, 2048],
      "block1.proj"           => [1, 1280, 2048],
      "block1.residual1"      => [1, 1280, 2048],
      "block1.norm2"          => [1, 1280, 2048],
      "block1.mlp_fc1"        => [1, 1280, 8192],
      "block1.mlp_silu"       => [1, 1280, 8192],
      "block1.mlp_fc2"        => [1, 1280, 2048],
      "block1.output"         => [1, 1280, 2048],
    }

    expected_shapes.each do |name, shape|
      dtype, actual_shape = HyNomasReferenceRefinerBlocksTrace.tensor_spec(name)
      dtype.should eq("F32")
      actual_shape.should eq(shape)
    end

    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.tensor_spec("unknown")
    end
  end

  it "requires the complete connected prelude BF16 control chain before block capture" do
    expect_raises(ArgumentError, "requires all five prelude BF16 controls") do
      HyNomasReferenceRefinerBlocksTrace.validate_prelude_controls!(true, true, true, true, false)
    end
    HyNomasReferenceRefinerBlocksTrace.validate_prelude_controls!(true, true, true, true, true).should be_nil
  end

  it "requires allocation-free encode and exactly one terminal drain" do
    HyNomasReferenceRefinerBlocksTrace.validate_execution!(0_i64, 0_i64, 0, 1).should be_nil

    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.validate_execution!(1_i64, 0_i64, 0, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.validate_execution!(0_i64, 1_i64, 0, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.validate_execution!(0_i64, 0_i64, 1, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.validate_execution!(0_i64, 0_i64, 0, 2)
    end
  end

  it "fails closed when required checkpoints are missing" do
    expect_raises(ArgumentError) do
      HyNomasReferenceRefinerBlocksTrace.tensor_sources({} of String => ML::MetalBuffer)
    end
  end

  it "declares native input sources and the single stored inter-block alias" do
    HyNomasReferenceRefinerBlocksTrace.checkpoint_binding("block0.input").should eq({"input_embed", nil})
    HyNomasReferenceRefinerBlocksTrace.checkpoint_binding("block0.output").should eq({"block0.output", nil})
    HyNomasReferenceRefinerBlocksTrace.checkpoint_binding("block1.input").should eq({"block0.output", "block0.output"})
  end

  it "requires fresh output paths distinct from both existing exports" do
    root = File.tempname("hy-nomas-refiner-blocks-trace")
    Dir.mkdir(root)
    begin
      cache_manifest, cache_payload = HyNomasReferenceCacheProbe.checked_output_paths!(
        File.join(root, "cache.json"))
      prelude_manifest, prelude_payload = HyNomasReferenceRefinerTrace.checked_output_paths!(
        File.join(root, "prelude.json"))
      blocks_manifest, blocks_payload = HyNomasReferenceRefinerBlocksTrace.checked_output_paths!(
        File.join(root, "blocks.json"))
      HyNomasReferenceRefinerBlocksTrace.ensure_distinct_outputs!(
        cache_manifest, cache_payload, prelude_manifest, prelude_payload,
        blocks_manifest, blocks_payload).should be_nil
      HyNomasReferenceRefinerBlocksTrace.ensure_fresh_outputs!(blocks_manifest, blocks_payload).should be_nil

      expect_raises(ArgumentError) do
        HyNomasReferenceRefinerBlocksTrace.ensure_distinct_outputs!(
          cache_manifest, cache_payload, prelude_manifest, prelude_payload,
          cache_manifest, blocks_payload)
      end

      File.write(blocks_payload, "existing owner")
      expect_raises(ArgumentError) do
        HyNomasReferenceRefinerBlocksTrace.ensure_fresh_outputs!(blocks_manifest, blocks_payload)
      end
      File.read(blocks_payload).should eq("existing owner")
    ensure
      Dir.glob(File.join(root, "*")).each { |path| File.delete(path) }
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "pins the exact compiled native trace source set" do
    pins = HyNomasReferenceRefinerBlocksTrace.compiled_source_sha256s
    pins.keys.should eq(HyNomasReferenceRefinerBlocksTrace::SOURCE_FILES)
    pins.values.each do |sha|
      HyNomasReferenceRefinerBlocksTrace.valid_sha256?(sha).should be_true
    end
  end
end
