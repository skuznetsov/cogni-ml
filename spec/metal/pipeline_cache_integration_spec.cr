require "spec"
require "../../src/ml/metal/device"

private def integration_key(
  owner : String,
  source : String = "kernel void integration_probe() {}",
) : ML::Metal::PipelineSpecializationKey
  ML::Metal::PipelineSpecializationKey.for_source(
    owner: owner,
    function_name: "integration_probe",
    source: source,
    compiler_abi: "metal-safe-v1",
    device_family: "cpu-only-probe",
    code_variant: "f16-h64-tile32"
  )
end

describe ML::Metal::PipelineCache do
  it "exposes bounded process admission without constructing a Metal pipeline" do
    before = ML::Metal::PipelineCache.typed_process_diagnostics
    owner = "spec-failed-build"
    key = integration_key(owner)
    source = "kernel void integration_probe() {}"
    ML::Metal::PipelineCache.register_typed_owner(owner, 2)

    expect_raises(Exception, "Metal disabled (cpu_only)") do
      ML::Metal::PipelineCache.get_or_compile(key, 2, source)
    end

    diagnostics = ML::Metal::PipelineCache.typed_diagnostics(owner).not_nil!
    diagnostics.capacity.should eq(2)
    diagnostics.entries.should eq(0)
    diagnostics.in_flight.should eq(0)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.compile_failures.should eq(1)

    after_probe = ML::Metal::PipelineCache.typed_process_diagnostics
    after_probe.owners.should eq(before.owners + 1)
    after_probe.declared_capacity.should eq(before.declared_capacity + 2)
    after_probe.max_owners.should eq(64)
    after_probe.max_declared_capacity.should eq(4096)
  end

  it "validates source identity before admitting an owner" do
    before = ML::Metal::PipelineCache.typed_process_diagnostics
    owner = "spec-source-mismatch"
    key = integration_key(owner)

    expect_raises(ArgumentError, /source digest mismatch/) do
      ML::Metal::PipelineCache.get_or_compile(
        key,
        1,
        "kernel void changed_probe() {}"
      )
    end

    ML::Metal::PipelineCache.typed_diagnostics(owner).should be_nil
    ML::Metal::PipelineCache.typed_process_diagnostics.should eq(before)
  end

  it "rejects a process capacity request before admitting an owner" do
    before = ML::Metal::PipelineCache.typed_process_diagnostics
    owner = "spec-process-capacity"
    expect_raises(ML::Metal::PipelineCacheCapacityError, /4096/) do
      ML::Metal::PipelineCache.register_typed_owner(owner, 4097)
    end

    ML::Metal::PipelineCache.typed_diagnostics(owner).should be_nil
    ML::Metal::PipelineCache.typed_process_diagnostics.should eq(before)
  end

  it "refuses to reinterpret an admitted owner with a different capacity" do
    owner = "spec-owner-capacity"
    ML::Metal::PipelineCache.register_typed_owner(owner, 2)

    expect_raises(ArgumentError, /already 2, not 3/) do
      ML::Metal::PipelineCache.register_typed_owner(owner, 3)
    end

    ML::Metal::PipelineCache.typed_diagnostics(owner).not_nil!.capacity.should eq(2)
  end

  it "refuses compilation for an owner that was not explicitly registered" do
    owner = "spec-unregistered-owner"
    key = integration_key(owner)

    expect_raises(ArgumentError, /must be registered/) do
      ML::Metal::PipelineCache.get_or_compile(
        key,
        1,
        "kernel void integration_probe() {}"
      )
    end

    ML::Metal::PipelineCache.typed_diagnostics(owner).should be_nil
  end
end
