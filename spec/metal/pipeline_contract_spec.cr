require "spec"
require "../../src/ml/metal/pipeline_contract"

private def pipeline_key(
  owner : String = "trellis2-dino",
  function_name : String = "dino_projection",
  source : String = "kernel void dino_projection() {}",
  code_variant : String = "f16-h64-tile32",
) : ML::Metal::PipelineSpecializationKey
  ML::Metal::PipelineSpecializationKey.for_source(
    owner: owner,
    function_name: function_name,
    source: source,
    compiler_abi: "metal-safe-v1",
    device_family: "apple-m2-max",
    code_variant: code_variant
  )
end

describe ML::Metal::PipelineSpecializationKey do
  it "binds source, compiler, device, and code specialization without runtime lengths" do
    first = pipeline_key
    second = pipeline_key

    first.should eq(second)
    first.source_digest.should eq(
      Digest::SHA256.hexdigest("kernel void dino_projection() {}")
    )
    first.canonical.should contain("metal-pipeline/v1/owner-trellis2-dino")
    first.canonical.should contain("fn-dino_projection")
    first.canonical.should contain("variant-f16-h64-tile32")
    first.canonical.should_not contain("tokens")
    first.canonical.should_not contain("1056")
  end

  it "rejects malformed cache identities before admission" do
    expect_raises(ArgumentError, /owner/) do
      pipeline_key(owner: "TRELLIS 2")
    end
    expect_raises(ArgumentError, /function_name/) do
      pipeline_key(function_name: "bad function")
    end
    expect_raises(ArgumentError, /source_digest/) do
      ML::Metal::PipelineSpecializationKey.new(
        owner: "trellis2-dino",
        function_name: "dino_projection",
        source_digest: "not-a-digest",
        compiler_abi: "metal-safe-v1",
        device_family: "apple-m2-max",
        code_variant: "f16-h64-tile32"
      )
    end
  end

  it "rejects source text that does not match the admitted digest" do
    expect_raises(ArgumentError, /source digest mismatch/) do
      pipeline_key.validate_source!("kernel void changed_projection() {}")
    end
  end

  it "snapshots caller strings and does not expose mutable hash identity" do
    owner = String.new("trellis2-dino".to_slice)
    key = pipeline_key(owner: owner)

    owner.to_unsafe[0] = 'x'.ord.to_u8
    key.owner.should eq("trellis2-dino")

    exposed = key.owner
    exposed.to_unsafe[0] = 'y'.ord.to_u8
    key.owner.should eq("trellis2-dino")
    key.should eq(pipeline_key)
  end
end

describe ML::Metal::BoundedPipelineCache do
  it "builds one artifact and reports a warm hit" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 2
    )
    key = pipeline_key
    builds = 0

    first = cache.fetch(key) do
      builds += 1
      "pipeline-#{builds}"
    end
    second = cache.fetch(key) do
      builds += 1
      "pipeline-#{builds}"
    end

    first.should eq("pipeline-1")
    second.should be(first)
    builds.should eq(1)
    cache.diagnostics.should eq(
      ML::Metal::PipelineCacheDiagnostics.new(
        owner: "trellis2-dino",
        capacity: 2,
        lookups: 2,
        hits: 1,
        misses: 1,
        compile_attempts: 1,
        compile_failures: 0,
        capacity_refusals: 0,
        in_flight_refusals: 0,
        entries: 1,
        in_flight: 0,
        high_water: 1
      )
    )
  end

  it "refuses capacity before invoking the builder" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 1
    )
    cache.fetch(pipeline_key) { "first" }
    invoked = false

    expect_raises(ML::Metal::PipelineCacheCapacityError, /capacity 1/) do
      cache.fetch(pipeline_key(function_name: "dino_attention")) do
        invoked = true
        "second"
      end
    end

    invoked.should be_false
    diagnostics = cache.diagnostics
    diagnostics.entries.should eq(1)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.capacity_refusals.should eq(1)
  end

  it "rolls back a failed build reservation and permits an exact retry" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 1
    )
    key = pipeline_key

    expect_raises(Exception, "compile failed") do
      cache.fetch(key) { raise "compile failed" }
    end
    failed = cache.diagnostics
    failed.entries.should eq(0)
    failed.in_flight.should eq(0)
    failed.compile_attempts.should eq(1)
    failed.compile_failures.should eq(1)

    cache.fetch(key) { "recovered" }.should eq("recovered")
    recovered = cache.diagnostics
    recovered.entries.should eq(1)
    recovered.compile_attempts.should eq(2)
  end

  it "fails closed on a same-key re-entrant build instead of double compiling" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 1
    )
    key = pipeline_key

    expect_raises(ML::Metal::PipelineCacheBuildInFlightError, /already in flight/) do
      cache.fetch(key) do
        cache.fetch(key) { "inner" }
      end
    end

    diagnostics = cache.diagnostics
    diagnostics.entries.should eq(0)
    diagnostics.in_flight.should eq(0)
    diagnostics.in_flight_refusals.should eq(1)
    diagnostics.compile_failures.should eq(1)
  end

  it "fails closed for a concurrent same-key caller while one build is in flight" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 1
    )
    key = pipeline_key
    started = Channel(Nil).new
    release = Channel(Nil).new
    completed = Channel(String).new

    spawn do
      completed.send(cache.fetch(key) do
        started.send(nil)
        release.receive
        "pipeline"
      end)
    end

    started.receive
    expect_raises(ML::Metal::PipelineCacheBuildInFlightError, /already in flight/) do
      cache.fetch(key) { "duplicate" }
    end
    release.send(nil)
    completed.receive.should eq("pipeline")

    diagnostics = cache.diagnostics
    diagnostics.entries.should eq(1)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.in_flight_refusals.should eq(1)
  end

  it "rejects keys from a different owner without observing the cache" do
    cache = ML::Metal::BoundedPipelineCache(String).new(
      owner: "trellis2-dino",
      capacity: 1
    )

    expect_raises(ArgumentError, /owned by trellis2-dino/) do
      cache.fetch(pipeline_key(owner: "trellis2-dense")) { "wrong" }
    end
    cache.diagnostics.lookups.should eq(0)
  end

  it "snapshots its owner independently of caller and diagnostic strings" do
    owner = String.new("trellis2-dino".to_slice)
    cache = ML::Metal::BoundedPipelineCache(String).new(owner, 1)
    owner.to_unsafe[0] = 'x'.ord.to_u8

    cache.owner.should eq("trellis2-dino")
    diagnostics = cache.diagnostics
    diagnostics.owner.to_unsafe[0] = 'y'.ord.to_u8
    cache.diagnostics.owner.should eq("trellis2-dino")
  end
end
