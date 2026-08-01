require "../../spec_helper"
require "../../../src/ml/three_d/trellis2/cache_adapter"

private def cache_adapter_contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract
  profile = ML::ThreeD::Trellis2::DenseStageResourceProfile.new(
    id: "dense-cache-tiny",
    in_channels: 4,
    model_channels: 16,
    context_channels: 12,
    out_channels: 6,
    num_heads: 2,
    mlp_hidden_channels: 32,
    frequency_dim: 8,
    batch_buckets: [1_i32],
    voxel_buckets: [8_i32, 27_i32],
    context_buckets: [2_i32, 4_i32]
  )
  abi = ML::ThreeD::Trellis2::DenseKernelABI.new(
    source_digest: "2" * 64,
    device_family: "cpu-oracle",
    compiler_abi: "t2n2d1a-v1",
    weight_format: "f32-reference",
    accumulation_dtype: ML::DType::F32,
    activation_mode: "silu-gelu",
    normalization_mode: "layer-rms",
    attention_mode: "self-cross-qkrms",
    rope_mode: "realpair-3d",
    mask_mode: "right-valid-trim",
    layout_mode: "ncdhw-cubic"
  )
  ML::ThreeD::Trellis2::DenseDeviceResourceContract.new(
    profiles: [profile],
    kernel_variants: ["mlp", "self-attention"],
    dtypes: [ML::DType::F32],
    cache_owner: ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU::CACHE_OWNER,
    kernel_abi: abi,
    max_axis_padding_ratio: 4.0,
    max_single_tensor_bytes: 1_i64 * 1024_i64 * 1024_i64,
    max_declared_activation_bytes: 8_i64 * 1024_i64 * 1024_i64
  )
end

private def cache_adapter_key(
  contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract,
  voxel_tokens : Int32 = 8,
  context_tokens : Int32 = 2,
) : ML::ThreeD::Trellis2::KernelSpecializationKey
  request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
    "dense-cache-tiny",
    1,
    voxel_tokens,
    context_tokens,
    ML::DType::F32
  )
  contract.plan(request).kernel_keys.first
end

private def cache_adapter(
  contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract,
  capacity : Int32 = 4,
  compile_calls : Array(String) = [] of String,
) : ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU
  ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(
    contract,
    contract.theoretical_kernel_key_count
  )
  compiler = ->(key : ML::ThreeD::Trellis2::KernelSpecializationKey) do
    compile_calls << key.canonical
    "compiled:#{key.canonical}"
  end
  ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU.new(
    ledger,
    capacity,
    compiler
  )
end

describe ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU do
  it "refuses foreign and undeclared keys before any cache observation" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, compile_calls: compile_calls)
    declared = cache_adapter_key(contract)
    foreign = declared.copy_with(cache_owner: "caller-owned")
    undeclared = declared.copy_with(kernel_variant: "undeclared")

    expect_raises(ArgumentError, /fixed cache owner/) { adapter.fetch!(foreign) }
    expect_raises(ArgumentError, /not declared/) { adapter.fetch!(undeclared) }

    adapter.diagnostics.should eq(
      ML::ThreeD::Trellis2::KernelCacheDiagnostics.new(0, 0, 0, 0, 0, 0)
    )
    compile_calls.should be_empty
  end

  it "cancels after admission but before lookup without touching the cache" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, compile_calls: compile_calls)
    checks = 0
    cancelled = -> do
      checks += 1
      checks >= 2
    end

    expect_raises(ML::ThreeD::Trellis2::KernelCacheCancelledError) do
      adapter.fetch!(cache_adapter_key(contract), cancelled)
    end
    checks.should eq(2)
    adapter.diagnostics.should eq(
      ML::ThreeD::Trellis2::KernelCacheDiagnostics.new(0, 0, 0, 0, 0, 0)
    )
    compile_calls.should be_empty
  end

  it "compiles one key once and returns a cache hit on repetition" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, compile_calls: compile_calls)
    key = cache_adapter_key(contract)

    first = adapter.fetch!(key)
    second = adapter.fetch!(key)

    first.should eq(second)
    compile_calls.should eq([key.canonical])
    adapter.lifecycle.should eq("adapter-retained-no-eviction")
    adapter.diagnostics.should eq(
      ML::ThreeD::Trellis2::KernelCacheDiagnostics.new(2, 1, 1, 1, 0, 1)
    )
  end

  it "compiles with the ledger-owned canonical key" do
    contract = cache_adapter_contract
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(
      contract,
      contract.theoretical_kernel_key_count
    )
    key = cache_adapter_key(contract)
    canonical = ledger.admit!(key)
    equal_key = key.copy_with(
      profile_signature: String.build { |io| io << key.profile_signature }
    )
    compiled_keys = [] of ML::ThreeD::Trellis2::KernelSpecializationKey
    adapter = ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU.new(
      ledger,
      1,
      ->(candidate : ML::ThreeD::Trellis2::KernelSpecializationKey) do
        compiled_keys << candidate
        "compiled:#{candidate.canonical}"
      end
    )

    equal_key.profile_signature.same?(canonical.profile_signature).should be_false
    first = adapter.fetch!(equal_key)
    second = adapter.fetch!(key)
    first.same?(second).should be_true
    compiled_keys.size.should eq(1)
    compiled_keys.first.profile_signature.same?(canonical.profile_signature).should be_true
    adapter.diagnostics.should eq(
      ML::ThreeD::Trellis2::KernelCacheDiagnostics.new(2, 1, 1, 1, 0, 1)
    )
  end

  it "compiles a contended key once" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, compile_calls: compile_calls)
    key = cache_adapter_key(contract)
    workers = 32
    ready = Channel(Nil).new(workers)
    release = Channel(Nil).new(workers)
    results = Channel(String).new(workers)

    workers.times do
      spawn do
        ready.send(nil)
        release.receive
        results.send(adapter.fetch!(key))
      end
    end
    workers.times { ready.receive }
    workers.times { release.send(nil) }
    values = Array.new(workers) { results.receive }

    values.uniq.size.should eq(1)
    compile_calls.should eq([key.canonical])
    diagnostics = adapter.diagnostics
    diagnostics.lookups.should eq(workers)
    diagnostics.hits.should eq(workers - 1)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.entries.should eq(1)
  end

  it "keeps full structured keys distinct" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, compile_calls: compile_calls)
    first = cache_adapter_key(contract, voxel_tokens: 8, context_tokens: 2)
    second = cache_adapter_key(contract, voxel_tokens: 27, context_tokens: 4)

    adapter.fetch!(first).should_not eq(adapter.fetch!(second))
    compile_calls.sort.should eq([first.canonical, second.canonical].sort)
    adapter.diagnostics.entries.should eq(2)
  end

  it "refuses a new cache entry at capacity before compilation" do
    contract = cache_adapter_contract
    compile_calls = [] of String
    adapter = cache_adapter(contract, capacity: 1, compile_calls: compile_calls)
    first = cache_adapter_key(contract, voxel_tokens: 8, context_tokens: 2)
    second = cache_adapter_key(contract, voxel_tokens: 27, context_tokens: 4)

    adapter.fetch!(first)
    expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /adapter capacity 1/) do
      adapter.fetch!(second)
    end

    compile_calls.should eq([first.canonical])
    adapter.diagnostics.should eq(
      ML::ThreeD::Trellis2::KernelCacheDiagnostics.new(2, 0, 2, 1, 1, 1)
    )
  end
end
