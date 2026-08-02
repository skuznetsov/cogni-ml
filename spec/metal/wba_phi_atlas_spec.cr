require "spec"
require "../../src/ml/metal/wba_phi_atlas"

private def resource_envelope(
  pipeline_entries : Int32 = 8,
  pipeline_capacity : Int32 = 12,
  pipeline_compile_attempts : Int64 = 8_i64,
  pipeline_compile_budget : Int64 = 12_i64,
  peak_live_bytes : Int64 = 800_i64,
  peak_live_budget_bytes : Int64 = 1_000_i64,
  memory_pressure_events : Int32 = 0,
  padding_waste_ppm : Int32 = 26_000,
  padding_waste_limit_ppm : Int32 = 50_000,
) : ML::Metal::Wba::ResourceEnvelope
  ML::Metal::Wba::ResourceEnvelope.new(
    pipeline_entries: pipeline_entries,
    pipeline_capacity: pipeline_capacity,
    pipeline_compile_attempts: pipeline_compile_attempts,
    pipeline_compile_budget: pipeline_compile_budget,
    peak_live_bytes: peak_live_bytes,
    peak_live_budget_bytes: peak_live_budget_bytes,
    memory_pressure_events: memory_pressure_events,
    padding_waste_ppm: padding_waste_ppm,
    padding_waste_limit_ppm: padding_waste_limit_ppm
  )
end

private def wba_card : ML::Metal::Wba::Card
  ML::Metal::Wba::Card.new(
    window_or_trigger: "dino-transformer-blocks",
    transport_corridor: "padded-token-bucket",
    legal_move: "reuse-one-code-specialized-pipeline",
    boundary_safety: "exact-shape-parity-and-capacity-gates",
    recompute_safety: "paired-full-corridor-wall-and-potential",
    dual_frame: "gpu-active-window-plus-end-to-end-wall",
    local_certificate: "pipeline-key-and-paired-samples"
  )
end

private def corridor_observation(
  bucket_id : String = "tokens-1056",
  active_window_gpu_ns : Int64 = 10_000_000_i64,
  wall_ns : Int64 = 12_000_000_i64,
  parity_failures : Int32 = 0,
  sync_count : Int32 = 6,
  resources : ML::Metal::Wba::ResourceEnvelope = resource_envelope,
  active_window : String = "transformer-blocks",
) : ML::Metal::Wba::CorridorObservation
  ML::Metal::Wba::CorridorObservation.new(
    card: wba_card,
    corridor_id: "dino-encoder",
    profile_id: "dinov3-vitl16-512",
    bucket_id: bucket_id,
    active_window: active_window,
    active_window_gpu_ns: active_window_gpu_ns,
    wall_ns: wall_ns,
    tied_dominant_routes: 1,
    sync_count: sync_count,
    boundary_transfer_bytes: 4_096_i64,
    remaining_work_bytes: 1_000_000_i64,
    parity_failures: parity_failures,
    resources: resources
  )
end

describe ML::Metal::Wba::Atlas do
  it "keeps an unpaired profile at profile-only" do
    analysis = ML::Metal::Wba::Atlas.profile(corridor_observation)

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::ProfileOnly)
    analysis.reasons.should contain("missing_paired_certificate")
  end

  it "uses a known positive control to classify bounded full-corridor descent" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64,
      sync_count: 4
    )
    certificate = ML::Metal::Wba::PairedCertificate.new(
      pairs: 7,
      candidate_wins: 7
    )

    analysis = ML::Metal::Wba::Atlas.compare(baseline, candidate, certificate)

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::CandidateDescent)
    analysis.candidate.not_nil!.potential.descends_from?(baseline.potential).should be_true
    analysis.reasons.should contain("paired_full_corridor_descent")
  end

  it "refutes a faster kernel when the bounded pipeline capacity is exceeded" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64,
      resources: resource_envelope(pipeline_entries: 13)
    )
    certificate = ML::Metal::Wba::PairedCertificate.new(7, 7)

    analysis = ML::Metal::Wba::Atlas.compare(baseline, candidate, certificate)

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Refuted)
    analysis.reasons.should contain("pipeline_capacity_exceeded")
  end

  it "refutes compile churn even when live pipeline cardinality stays bounded" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64,
      resources: resource_envelope(pipeline_compile_attempts: 13)
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Refuted)
    analysis.reasons.should contain("pipeline_compile_budget_exceeded")
  end

  it "refutes local GPU descent when recomputed full-corridor wall regresses" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 13_000_000_i64
    )
    certificate = ML::Metal::Wba::PairedCertificate.new(7, 2)

    analysis = ML::Metal::Wba::Atlas.compare(baseline, candidate, certificate)

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Refuted)
    analysis.reasons.should contain("full_corridor_wall_did_not_descend")
  end

  it "refuses to compare certificates from different padded buckets" do
    baseline = corridor_observation(bucket_id: "tokens-1056")
    candidate = corridor_observation(
      bucket_id: "tokens-4128",
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Incomparable)
    analysis.reasons.should contain("corridor_identity_mismatch")
  end

  it "refuses to compare different GPU timing windows" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window: "projection-only",
      active_window_gpu_ns: 1_000_000_i64,
      wall_ns: 10_000_000_i64
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Incomparable)
    analysis.reasons.should contain("corridor_identity_mismatch")
  end

  it "refuses a candidate that moves its own resource-policy goalposts" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64,
      resources: resource_envelope(peak_live_budget_bytes: 2_000_i64)
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Incomparable)
    analysis.reasons.should contain("resource_policy_mismatch")
  end

  it "requires enough paired observations before admitting a candidate" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(1, 1)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::InsufficientEvidence)
    analysis.reasons.should contain("at_least_three_pairs_required")
  end

  it "does not interpret missing GPU timestamps as perfect timing" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 0_i64,
      wall_ns: 10_000_000_i64
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::InsufficientEvidence)
    analysis.reasons.should contain("positive_gpu_timing_required")
  end

  it "does not interpret a missing wall-clock sample as full-corridor descent" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 0_i64
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::InsufficientEvidence)
    analysis.reasons.should contain("positive_full_corridor_wall_required")
  end

  it "requires a complete LTP/WBA corridor card" do
    expect_raises(ArgumentError, /recompute_safety/) do
      ML::Metal::Wba::Card.new(
        window_or_trigger: "dino-transformer-blocks",
        transport_corridor: "padded-token-bucket",
        legal_move: "reuse-one-code-specialized-pipeline",
        boundary_safety: "exact-shape-parity-and-capacity-gates",
        recompute_safety: "",
        dual_frame: "gpu-active-window-plus-end-to-end-wall",
        local_certificate: "pipeline-key-and-paired-samples"
      )
    end
  end

  it "rejects impossible padding-waste ppm policies" do
    expect_raises(ArgumentError, /must not exceed 1000000/) do
      resource_envelope(padding_waste_limit_ppm: 1_000_001)
    end
  end

  it "uses a seeded parity defect to qualify the negative gate" do
    baseline = corridor_observation
    candidate = corridor_observation(
      active_window_gpu_ns: 8_000_000_i64,
      wall_ns: 10_000_000_i64,
      parity_failures: 1
    )

    analysis = ML::Metal::Wba::Atlas.compare(
      baseline,
      candidate,
      ML::Metal::Wba::PairedCertificate.new(7, 7)
    )

    analysis.verdict.should eq(ML::Metal::Wba::Verdict::Refuted)
    analysis.reasons.should contain("parity_failures=1")
  end
end
