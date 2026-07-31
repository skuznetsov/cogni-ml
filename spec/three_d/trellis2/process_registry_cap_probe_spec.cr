require "../../spec_helper"

# These probes intentionally run only when selected explicitly. Each exhausts
# process-lifetime registry state, so including it in the ordinary randomized
# TRELLIS suite would make unrelated examples order-dependent.

private def registry_cap_probe_abi : ML::ThreeD::Trellis2::DenseKernelABI
  ML::ThreeD::Trellis2::DenseKernelABI.new(
    source_digest: "b" * 64,
    device_family: "cpu-oracle",
    compiler_abi: "t2n2d0-v1",
    weight_format: "f32-reference",
    accumulation_dtype: ML::DType::F32,
    activation_mode: "silu-gelu",
    normalization_mode: "layer-rms",
    attention_mode: "self-cross-qkrms",
    rope_mode: "realpair-3d",
    mask_mode: "right-valid-trim",
    layout_mode: "ncdhw-cubic"
  )
end

private def registry_cap_probe_contract(
  owner : String,
  wide : Bool = false,
) : ML::ThreeD::Trellis2::DenseDeviceResourceContract
  buckets = wide ? (1..16).map(&.to_i32) : [1_i32]
  voxel_buckets = buckets.map { |value| value * value * value }
  profile = ML::ThreeD::Trellis2::DenseStageResourceProfile.new(
    id: "registry-cap-probe",
    in_channels: 1,
    model_channels: 6,
    context_channels: 1,
    out_channels: 1,
    num_heads: 1,
    mlp_hidden_channels: 6,
    frequency_dim: 2,
    batch_buckets: buckets,
    voxel_buckets: voxel_buckets,
    context_buckets: buckets
  )
  ML::ThreeD::Trellis2::DenseDeviceResourceContract.new(
    profiles: [profile],
    kernel_variants: ["mlp"],
    dtypes: [ML::DType::F32],
    cache_owner: owner,
    kernel_abi: registry_cap_probe_abi,
    max_axis_padding_ratio: 16.0,
    max_single_tensor_bytes: 1_i64 << 36,
    max_declared_activation_bytes: 1_i64 << 40
  )
end

{% if flag?(:registry_owner_cap_probe) %}
  describe "T2N2d0b process owner ceiling" do
    it "refuses the sixty-fifth process owner" do
      ML::ThreeD::Trellis2::BoundedKernelKeyLedger::MAX_PROCESS_OWNERS.times do |index|
        contract = registry_cap_probe_contract("probe-owner-#{index}")
        ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
      end

      overflow = registry_cap_probe_contract("probe-owner-overflow")
      expect_raises(ML::ThreeD::Trellis2::KernelCacheCapacityError, /owner capacity 64/) do
        ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(overflow, 1)
      end
    end
  end
{% end %}

{% if flag?(:registry_key_cap_probe) %}
  describe "T2N2d0b process key ceiling" do
    it "refuses the four-thousand-and-ninety-seventh process key" do
      full = registry_cap_probe_contract("probe-key-full", wide: true)
      full.theoretical_kernel_key_count.should eq(4096)
      ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(full, 4096)
      ledger.admit!(full.all_kernel_keys)
      ledger.size.should eq(4096)

      overflow = registry_cap_probe_contract("probe-key-overflow")
      overflow_ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(overflow, 1)
      expect_raises(
        ML::ThreeD::Trellis2::KernelCacheCapacityError,
        /process kernel key capacity 4096/
      ) do
        overflow_ledger.admit!(overflow.all_kernel_keys)
      end
      overflow_ledger.size.should eq(0)
      ledger.size.should eq(4096)
    end
  end
{% end %}
