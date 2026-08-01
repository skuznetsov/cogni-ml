require "../../spec_helper"

private def dense_resource_profile(
  id : String = "dense-flow-tiny",
  model_channels : Int32 = 16,
  num_heads : Int32 = 2,
  frequency_dim : Int32 = 8,
  batch_buckets : Array(Int32) = [1_i32, 2_i32],
  voxel_buckets : Array(Int32) = [8_i32, 64_i32],
  context_buckets : Array(Int32) = [4_i32, 8_i32],
) : ML::ThreeD::Trellis2::DenseStageResourceProfile
  ML::ThreeD::Trellis2::DenseStageResourceProfile.new(
    id: id,
    in_channels: 4,
    model_channels: model_channels,
    context_channels: 12,
    out_channels: 6,
    num_heads: num_heads,
    mlp_hidden_channels: 32,
    frequency_dim: frequency_dim,
    batch_buckets: batch_buckets,
    voxel_buckets: voxel_buckets,
    context_buckets: context_buckets
  )
end

private def dense_kernel_abi(
  source_digest : String = "1" * 64,
  device_family : String = "unbound-device",
  accumulation_dtype : ML::DType = ML::DType::F32,
  rope_mode : String = "realpair-3d",
  mask_mode : String = "right-valid-trim",
  layout_mode : String = "ncdhw-cubic",
) : ML::ThreeD::Trellis2::DenseKernelABI
  ML::ThreeD::Trellis2::DenseKernelABI.new(
    source_digest: source_digest,
    device_family: device_family,
    compiler_abi: "t2n2d0-v1",
    weight_format: "unbound-reference",
    accumulation_dtype: accumulation_dtype,
    activation_mode: "silu-gelu",
    normalization_mode: "layer-rms",
    attention_mode: "self-cross-qkrms",
    rope_mode: rope_mode,
    mask_mode: mask_mode,
    layout_mode: layout_mode
  )
end

private def dense_resource_contract(
  profiles : Array(ML::ThreeD::Trellis2::DenseStageResourceProfile) = [dense_resource_profile],
  variants : Array(String) = ["self-attention", "cross-attention", "mlp"],
  dtypes : Array(ML::DType) = [ML::DType::F32, ML::DType::BF16],
  cache_owner : String = "dense-flow-owner",
  kernel_abi : ML::ThreeD::Trellis2::DenseKernelABI = dense_kernel_abi,
  max_axis_padding_ratio : Float64 = 8.0,
  max_single_tensor_bytes : Int64 = 1_i64 * 1024_i64 * 1024_i64,
  max_declared_activation_bytes : Int64 = 8_i64 * 1024_i64 * 1024_i64,
) : ML::ThreeD::Trellis2::DenseDeviceResourceContract
  ML::ThreeD::Trellis2::DenseDeviceResourceContract.new(
    profiles: profiles,
    kernel_variants: variants,
    dtypes: dtypes,
    cache_owner: cache_owner,
    kernel_abi: kernel_abi,
    max_axis_padding_ratio: max_axis_padding_ratio,
    max_single_tensor_bytes: max_single_tensor_bytes,
    max_declared_activation_bytes: max_declared_activation_bytes
  )
end

private def dense_resource_request(
  batch : Int32 = 1,
  voxel_tokens : Int32 = 27,
  context_tokens : Int32 = 3,
  dtype : ML::DType = ML::DType::F32,
  profile_id : String = "dense-flow-tiny",
) : ML::ThreeD::Trellis2::DenseActivationRequest
  ML::ThreeD::Trellis2::DenseActivationRequest.new(
    profile_id,
    batch,
    voxel_tokens,
    context_tokens,
    dtype
  )
end

describe ML::ThreeD::Trellis2::DenseDeviceResourceContract do
  it "exposes finite process-owner and aggregate-key ceilings" do
    ML::ThreeD::Trellis2::BoundedKernelKeyLedger::MAX_PROCESS_OWNERS.should eq(64)
    ML::ThreeD::Trellis2::BoundedKernelKeyLedger::MAX_PROCESS_KERNEL_KEYS.should eq(4096)
  end

  it "separates logical shapes from finite padded specialization keys" do
    contract = dense_resource_contract
    first = contract.plan(dense_resource_request(batch: 1, voxel_tokens: 27, context_tokens: 3))
    second = contract.plan(dense_resource_request(batch: 1, voxel_tokens: 64, context_tokens: 4))

    first.logical_batch.should eq(1)
    first.logical_voxel_tokens.should eq(27)
    first.logical_context_tokens.should eq(3)
    first.padded_batch.should eq(1)
    first.padded_voxel_tokens.should eq(64)
    first.padded_context_tokens.should eq(4)
    first.padding_policy.should eq("right-zero-mask-trim-v1")
    first.requires_attention_mask?.should be_true
    first.requires_output_trim?.should be_true
    first.rope_padding_phase.should eq({1.0, 0.0})
    first.kernel_keys.should eq(second.kernel_keys)
    abi = dense_kernel_abi.canonical
    prefix = "trellis2/resource-v1/owner-dense-flow-owner/#{abi}/" \
             "dense-flow-tiny/ncdhw-cubic-i4-c16-k12-o6-h2-m32-f8"
    first.kernel_keys.map(&.canonical).should eq([
      "#{prefix}/self-attention/f32/right-zero-mask-trim-v1/b1/n64/s4",
      "#{prefix}/cross-attention/f32/right-zero-mask-trim-v1/b1/n64/s4",
      "#{prefix}/mlp/f32/right-zero-mask-trim-v1/b1/n64/s4",
    ])

    current_fixture = contract.plan(
      dense_resource_request(batch: 2, voxel_tokens: 8, context_tokens: 3)
    )
    current_fixture.padded_batch.should eq(2)
    current_fixture.padded_voxel_tokens.should eq(8)
    current_fixture.padded_context_tokens.should eq(4)
  end

  it "enumerates the exact declared Cartesian key space" do
    contract = dense_resource_contract

    contract.theoretical_kernel_key_count.should eq(48)
    keys = contract.all_kernel_keys
    keys.size.should eq(48)
    keys.uniq.size.should eq(48)
    keys.all? { |key| contract.allows?(key) }.should be_true
  end

  it "bounds arbitrary logical lengths by the declared key space" do
    contract = dense_resource_contract(
      dtypes: [ML::DType::F32],
      max_axis_padding_ratio: 16.0
    )
    keys = Set(ML::ThreeD::Trellis2::KernelSpecializationKey).new

    [1_i32, 8_i32, 27_i32, 64_i32].each do |voxel_tokens|
      (1..8).each do |context_tokens|
        plan = contract.plan(
          dense_resource_request(
            voxel_tokens: voxel_tokens,
            context_tokens: context_tokens
          )
        )
        plan.kernel_keys.each { |key| keys << key }
      end
    end

    keys.size.should eq(12)
    keys.size.should be <= contract.theoretical_kernel_key_count
  end

  it "keeps 10x hostile spatial and context lengths inside refusal or the finite space" do
    contract = dense_resource_contract(
      variants: ["self-attention", "cross-attention"],
      dtypes: [ML::DType::F32],
      max_axis_padding_ratio: 16.0
    )
    admitted = Set(ML::ThreeD::Trellis2::KernelSpecializationKey).new

    (1..40).each do |resolution|
      voxel_tokens = resolution * resolution * resolution
      if resolution <= 4
        contract.plan(dense_resource_request(voxel_tokens: voxel_tokens)).kernel_keys.each do |key|
          admitted << key
        end
      else
        expect_raises(ArgumentError, /no declared voxel token bucket/) do
          contract.plan(dense_resource_request(voxel_tokens: voxel_tokens))
        end
      end
    end
    (1..80).each do |context_tokens|
      if context_tokens <= 8
        contract.plan(
          dense_resource_request(voxel_tokens: 8, context_tokens: context_tokens)
        ).kernel_keys.each { |key| admitted << key }
      else
        expect_raises(ArgumentError, /no declared context token bucket/) do
          contract.plan(
            dense_resource_request(voxel_tokens: 8, context_tokens: context_tokens)
          )
        end
      end
    end

    admitted.size.should be <= contract.theoretical_kernel_key_count
  end

  it "accounts the declared padded activation inventory without claiming runtime peak" do
    contract = dense_resource_contract(dtypes: [ML::DType::F32])
    plan = contract.plan(dense_resource_request)

    plan.declared_tensor_bytes["self attention scores"].should eq(32_768)
    plan.declared_tensor_bytes["cross attention scores"].should eq(2_048)
    plan.max_single_tensor_bytes.should eq(plan.declared_tensor_bytes.values.max)
    plan.declared_activation_bytes.should eq(plan.declared_tensor_bytes.values.sum)
    plan.declared_activation_bytes.should be <= contract.max_declared_activation_bytes
    bf16_plan = dense_resource_contract.plan(
      dense_resource_request(dtype: ML::DType::BF16)
    )
    bf16_plan.declared_tensor_bytes["self attention scores"].should eq(32_768)
    caller_copy = plan.declared_tensor_bytes
    caller_copy.clear
    plan.declared_tensor_bytes.should_not be_empty
  end

  it "copies caller-owned declarations before deriving cardinality" do
    batches = [1_i32, 2_i32]
    voxels = [8_i32, 64_i32]
    contexts = [4_i32, 8_i32]
    variants = ["self-attention", "mlp"]
    dtypes = [ML::DType::F32]
    profile = dense_resource_profile(
      batch_buckets: batches,
      voxel_buckets: voxels,
      context_buckets: contexts
    )
    contract = dense_resource_contract(
      profiles: [profile],
      variants: variants,
      dtypes: dtypes
    )

    batches << 4
    voxels << 125
    contexts << 16
    variants << "late-variant"
    dtypes << ML::DType::BF16

    contract.theoretical_kernel_key_count.should eq(16)
    contract.all_kernel_keys.size.should eq(16)
  end

  it "separates otherwise identical specialization spaces by owner and typed ABI" do
    first = dense_resource_contract
    second = dense_resource_contract(
      kernel_abi: dense_kernel_abi(device_family: "apple-family-nine")
    )
    third = dense_resource_contract(cache_owner: "second-owner")

    first_key = first.plan(dense_resource_request).kernel_keys.first
    second_key = second.plan(dense_resource_request).kernel_keys.first
    third_key = third.plan(dense_resource_request).kernel_keys.first
    first_key.should_not eq(second_key)
    first_key.canonical.should_not eq(second_key.canonical)
    first.allows?(second_key).should be_false
    first_key.should_not eq(third_key)
    first.allows?(third_key).should be_false
  end

  it "exposes the declared typed kernel ABI without canonical-string parsing" do
    abi = dense_kernel_abi(
      device_family: "typed-device",
      accumulation_dtype: ML::DType::BF16
    )
    contract = dense_resource_contract(kernel_abi: abi)

    contract.kernel_abi.source_digest.should eq("1" * 64)
    contract.kernel_abi.device_family.should eq("typed-device")
    contract.kernel_abi.compiler_abi.should eq("t2n2d0-v1")
    contract.kernel_abi.accumulation_dtype.should eq(ML::DType::BF16)
    contract.kernel_abi.mask_mode.should eq("right-valid-trim")
    contract.kernel_abi.rope_mode.should eq("realpair-3d")
    contract.kernel_abi.layout_mode.should eq("ncdhw-cubic")
    contract.kernel_abi.canonical.should eq(abi.canonical)
  end

  it "rejects malformed or unbounded declarations" do
    expect_raises(ArgumentError, /strictly increasing/) do
      dense_resource_profile(voxel_buckets: [8_i32, 8_i32])
    end
    expect_raises(ArgumentError, /strictly increasing/) do
      dense_resource_profile(context_buckets: [8_i32, 4_i32])
    end
    expect_raises(ArgumentError, /exact positive cubes/) do
      dense_resource_profile(voxel_buckets: [8_i32, 16_i32])
    end
    expect_raises(ArgumentError, /profile id/) do
      dense_resource_profile(id: "../dense")
    end
    expect_raises(ArgumentError, /3D RoPE frequency/) do
      dense_resource_profile(model_channels: 4, num_heads: 1)
    end
    expect_raises(ArgumentError, /6C modulation width overflow/) do
      dense_resource_profile(model_channels: Int32::MAX // 6 + 1, num_heads: 1)
    end
    expect_raises(ArgumentError, /frequency_dim must be at least 2/) do
      dense_resource_profile(frequency_dim: 1)
    end
    expect_raises(ArgumentError, /floating/) do
      dense_resource_contract(dtypes: [ML::DType::I32])
    end
    expect_raises(ArgumentError, /cache_owner/) do
      dense_resource_contract(cache_owner: "../metal")
    end
    expect_raises(ArgumentError, /source_digest/) do
      dense_kernel_abi(source_digest: "abc")
    end
    expect_raises(ArgumentError, /accumulation dtype/) do
      dense_kernel_abi(accumulation_dtype: ML::DType::I32)
    end
    expect_raises(ArgumentError, /mask_mode/) do
      dense_resource_contract(kernel_abi: dense_kernel_abi(mask_mode: "none"))
    end
    expect_raises(ArgumentError, /rope_mode/) do
      dense_resource_contract(kernel_abi: dense_kernel_abi(rope_mode: "none"))
    end
    expect_raises(ArgumentError, /layout_mode/) do
      dense_resource_contract(kernel_abi: dense_kernel_abi(layout_mode: "nhwc"))
    end
    expect_raises(ArgumentError, /duplicate kernel variant/) do
      dense_resource_contract(variants: ["mlp", "mlp"])
    end
    expect_raises(ArgumentError, /duplicate profile id/) do
      profile = dense_resource_profile
      dense_resource_contract(profiles: [profile, profile])
    end
    expect_raises(ArgumentError, /theoretical kernel key count/) do
      buckets = (1..16).map(&.to_i32)
      cubes = buckets.map { |value| value * value * value }
      profile = dense_resource_profile(
        batch_buckets: buckets,
        voxel_buckets: cubes,
        context_buckets: buckets
      )
      dense_resource_contract(
        profiles: [profile],
        variants: ["variant-a", "variant-b"],
        dtypes: [ML::DType::F32]
      )
    end
  end

  it "refuses unsupported, excessively padded, or over-budget requests before admission" do
    contract = dense_resource_contract(max_axis_padding_ratio: 2.0)
    expect_raises(ArgumentError, /positive/) do
      contract.plan(dense_resource_request(voxel_tokens: 0))
    end
    expect_raises(ArgumentError, /exact positive cube/) do
      contract.plan(dense_resource_request(voxel_tokens: 9))
    end
    expect_raises(ArgumentError, /no declared exact batch value/) do
      contract.plan(dense_resource_request(batch: 3))
    end
    expect_raises(ArgumentError, /no declared voxel token bucket/) do
      contract.plan(dense_resource_request(voxel_tokens: 125))
    end
    expect_raises(ArgumentError, /padding ratio/) do
      contract.plan(dense_resource_request(context_tokens: 1))
    end
    expect_raises(ArgumentError, /unsupported dtype/) do
      contract.plan(dense_resource_request(dtype: ML::DType::F16))
    end
    expect_raises(ArgumentError, /unknown dense resource profile/) do
      contract.plan(dense_resource_request(profile_id: "missing"))
    end

    tensor_limited = dense_resource_contract(max_single_tensor_bytes: 32_767)
    expect_raises(ArgumentError, /self attention scores.*single tensor budget/) do
      tensor_limited.plan(dense_resource_request)
    end
    trace_limited = dense_resource_contract(
      max_single_tensor_bytes: 32_768,
      max_declared_activation_bytes: 32_768
    )
    expect_raises(ArgumentError, /declared activation budget/) do
      trace_limited.plan(dense_resource_request)
    end

    huge_profile = dense_resource_profile(
      batch_buckets: [1_i32],
      voxel_buckets: [2_146_689_000_i32],
      context_buckets: [1_i32]
    )
    huge = dense_resource_contract(
      profiles: [huge_profile],
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      max_axis_padding_ratio: 1.0
    )
    expect_raises(ArgumentError, /single tensor budget/) do
      huge.plan(dense_resource_request(voxel_tokens: 2_146_689_000_i32, context_tokens: 1))
    end
  end

  it "admits keys idempotently and refuses capacity atomically without eviction" do
    contract = dense_resource_contract(
      variants: ["self-attention", "mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: "ledger-idempotence-owner"
    )
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    first_keys = contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys
    second_keys = contract.plan(
      dense_resource_request(voxel_tokens: 27, context_tokens: 3)
    ).kernel_keys

    ledger.admit!(first_keys)
    ledger.size.should eq(2)
    ledger.admit!(first_keys)
    ledger.size.should eq(2)

    expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /capacity 2/) do
      ledger.admit!(second_keys)
    end
    ledger.size.should eq(2)
    ledger.keys.should eq(first_keys)
  end

  it "rejects foreign keys without mutating the bounded ledger" do
    contract = dense_resource_contract(
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: "ledger-foreign-owner"
    )
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
    foreign = ML::ThreeD::Trellis2::KernelSpecializationKey.new(
      "dense-flow-owner",
      dense_kernel_abi.canonical,
      "dense-flow-tiny",
      "ncdhw-cubic-i4-c16-k12-o6-h2-m32-f8",
      "undeclared",
      ML::DType::F32,
      "right-zero-mask-trim-v1",
      1,
      64,
      4
    )

    expect_raises(ArgumentError, /not declared by this contract/) do
      ledger.admit!([foreign])
    end
    ledger.size.should eq(0)
  end

  it "reuses one contract-local canonical key without trusting owner-wide membership" do
    owner = "ledger-contract-local-owner"
    first_contract = dense_resource_contract(
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: owner
    )
    second_contract = dense_resource_contract(
      variants: ["self-attention"],
      dtypes: [ML::DType::F32],
      cache_owner: owner
    )
    first = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(first_contract, 2)
    second = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(second_contract, 2)
    first_key = first_contract.plan(
      dense_resource_request(voxel_tokens: 8)
    ).kernel_keys.first
    equal_key = first_key.copy_with(
      profile_signature: String.build { |io| io << first_key.profile_signature }
    )

    equal_key.profile_signature.same?(first_key.profile_signature).should be_false
    canonical = first.admit!(first_key)
    repeated = first.admit!(equal_key)
    canonical.should eq(first_key)
    repeated.should eq(first_key)
    repeated.profile_signature.same?(canonical.profile_signature).should be_true
    first.size.should eq(1)

    # A process-owner hit is not a contract certificate. The second ledger has
    # to validate this key against its own declared variants before caching it.
    expect_raises(ArgumentError, /not declared by this contract/) do
      second.admit!(first_key)
    end
    second.size.should eq(1)

    second_key = second_contract.plan(
      dense_resource_request(voxel_tokens: 8)
    ).kernel_keys.first
    second.admit!(second_key).should eq(second_key)
    first.size.should eq(2)
    second.size.should eq(2)
  end

  it "does not cache a refused single-key admission" do
    contract = dense_resource_contract(
      variants: ["mlp", "self-attention"],
      dtypes: [ML::DType::F32],
      cache_owner: "ledger-single-refusal-owner"
    )
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
    keys = contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys

    ledger.admit!(keys.first).should eq(keys.first)
    2.times do
      expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /capacity 1/) do
        ledger.admit!(keys.last)
      end
    end
    ledger.admit!(keys.first).should eq(keys.first)
    ledger.size.should eq(1)
    ledger.keys.should eq([keys.first])
  end

  it "shares one owner-wide capacity across independent ledgers" do
    contract = dense_resource_contract(
      variants: ["mlp", "self-attention"],
      dtypes: [ML::DType::F32],
      cache_owner: "registry-shared-owner"
    )
    first = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    second = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    first_keys = contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys
    second_keys = contract.plan(dense_resource_request(voxel_tokens: 27)).kernel_keys

    first.admit!(first_keys)
    second.admit!(first_keys)
    first.size.should eq(2)
    second.size.should eq(2)

    expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /capacity 2/) do
      second.admit!([second_keys.first])
    end
    first.size.should eq(2)
    second.size.should eq(2)
    first.keys.should eq(first_keys)
    second.keys.should eq(first_keys)
  end

  it "rejects a conflicting capacity for an already registered owner before admission" do
    contract = dense_resource_contract(
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: "registry-capacity-owner"
    )
    first = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
    key = contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys.first

    expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /capacity.*owner|owner.*capacity/) do
      ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    end
    first.admit!([key])
    first.size.should eq(1)
  end

  it "refuses an atomic multi-key batch when concurrent owner ledgers race" do
    contract = dense_resource_contract(
      variants: ["mlp", "self-attention"],
      dtypes: [ML::DType::F32],
      cache_owner: "registry-race-owner"
    )
    first = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    second = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 2)
    first_batch = contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys
    second_batch = contract.plan(dense_resource_request(voxel_tokens: 27)).kernel_keys
    ready = Channel(Nil).new(2)
    release = Channel(Nil).new(2)
    outcomes = Channel(String).new(2)

    spawn do
      ready.send(nil)
      release.receive
      begin
        first.admit!(first_batch)
        outcomes.send("winner")
      rescue ex : ML::ThreeD::Trellis2::KernelCacheCapacityError
        outcomes.send("capacity")
      rescue ex : Exception
        outcomes.send("unexpected:#{ex.class}:#{ex.message}")
      end
    end
    spawn do
      ready.send(nil)
      release.receive
      begin
        second.admit!(second_batch)
        outcomes.send("winner")
      rescue ex : ML::ThreeD::Trellis2::KernelCacheCapacityError
        outcomes.send("capacity")
      rescue ex : Exception
        outcomes.send("unexpected:#{ex.class}:#{ex.message}")
      end
    end

    2.times { ready.receive }
    2.times { release.send(nil) }
    [outcomes.receive, outcomes.receive].sort.should eq(["capacity", "winner"])
    first.size.should eq(2)
    second.size.should eq(2)
    (first.keys == first_batch || first.keys == second_batch).should be_true
  end

  it "counts distinct ABI and profile identities against one owner capacity" do
    owner = "registry-identity-owner"
    first_contract = dense_resource_contract(
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: owner
    )
    second_contract = dense_resource_contract(
      profiles: [dense_resource_profile(id: "dense-alt")],
      variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: owner,
      kernel_abi: dense_kernel_abi(device_family: "alternate-device")
    )
    first = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(first_contract, 2)
    second = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(second_contract, 2)
    first_key = first_contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys.first
    second_key = second_contract.plan(
      dense_resource_request(profile_id: "dense-alt", voxel_tokens: 8)
    ).kernel_keys.first

    first_key.cache_owner.should eq(second_key.cache_owner)
    first_key.kernel_abi.should_not eq(second_key.kernel_abi)
    first_key.profile_id.should_not eq(second_key.profile_id)
    first.admit!([first_key])
    second.admit!([second_key])
    first.size.should eq(2)
    second.keys.should eq([first_key, second_key])

    next_key = first_contract.plan(dense_resource_request(voxel_tokens: 27)).kernel_keys.first
    expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /capacity 2/) do
      first.admit!([next_key])
    end
    first.size.should eq(2)
  end

  it "keeps the owner registry bounded under one hundred contending admissions" do
    contract = dense_resource_contract(
      variants: ["mlp", "self-attention"],
      dtypes: [ML::DType::F32],
      cache_owner: "registry-contention-owner"
    )
    batches = [
      contract.plan(dense_resource_request(voxel_tokens: 8)).kernel_keys,
      contract.plan(dense_resource_request(voxel_tokens: 27)).kernel_keys,
    ]
    workers = 100
    ledgers = Array.new(workers) do
      ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 4)
    end
    seed_ready = Channel(Nil).new(2)
    seed_release = Channel(Nil).new(2)
    seed_outcomes = Channel(String).new(2)

    2.times do |index|
      spawn do
        seed_ready.send(nil)
        seed_release.receive
        begin
          ledgers[index].admit!(batches[index])
          seed_outcomes.send("winner")
        rescue ex : ML::ThreeD::Trellis2::KernelCacheCapacityError
          seed_outcomes.send("capacity")
        rescue ex : Exception
          seed_outcomes.send("unexpected:#{ex.class}:#{ex.message}")
        end
      end
    end

    2.times { seed_ready.receive }
    2.times { seed_release.send(nil) }
    [seed_outcomes.receive, seed_outcomes.receive].sort.should eq(["winner", "winner"])

    overflow = contract.plan(
      dense_resource_request(voxel_tokens: 8, context_tokens: 8)
    ).kernel_keys
    remaining = workers - 2
    ready = Channel(Nil).new(remaining)
    release = Channel(Nil).new(remaining)
    outcomes = Channel(String).new(remaining)

    (2...workers).each do |index|
      spawn do
        ready.send(nil)
        release.receive
        begin
          # Even-indexed ledgers repeat an admitted batch; odd-indexed ledgers
          # race an all-new batch and must refuse at the owner-wide bound.
          candidate = index.even? ? batches[0] : overflow
          ledgers[index].admit!(candidate)
          outcomes.send("winner")
        rescue ex : ML::ThreeD::Trellis2::KernelCacheCapacityError
          outcomes.send("capacity")
        rescue ex : Exception
          outcomes.send("unexpected:#{ex.class}:#{ex.message}")
        end
      end
    end

    remaining.times { ready.receive }
    remaining.times { release.send(nil) }
    remaining_results = Array.new(remaining) { outcomes.receive }
    remaining_results.all? { |result| result == "winner" || result == "capacity" }.should be_true
    remaining_results.count("capacity").should eq(remaining // 2)

    expected = batches.flatten.uniq.map(&.canonical).sort
    expected.size.should be <= ML::ThreeD::Trellis2::BoundedKernelKeyLedger::MAX_PROCESS_KERNEL_KEYS
    ledgers.each do |ledger|
      ledger.size.should eq(expected.size)
      ledger.keys.map(&.canonical).sort.should eq(expected)
    end
  end
end
