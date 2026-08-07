# Opt-in fresh-process Darwin CPU measurement for the pinned hidden-1024
# DINOv3 embedding projection at 512x512. This is not a full encoder run.
# The compile-time flag exposes only the bounded manual probe path; the normal
# EmbeddingCPU forward keeps its production 64M-MAC limit.

{% if flag?(:dinov3_model_scale_forward_probe) %}
{% unless flag?(:darwin) %}
  {% raise "DINOv3 model-scale forward probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "../../spec_helper"
require "../../../src/ml/vision/dino_v3"

module DinoV3ModelScaleForwardProbe
  extend self

  PROBE_EDGE              = 512_i32
  EXPECTED_FILE_BYTES     = 1_212_559_808_i64
  EXPECTED_PARAMETER_BYTES = 3_170_304_i64
  EXPECTED_PARAMETER_SHA256 = "8124dfb9859ecbb6d33908bcd013e32a469897140cb413c2aa5b629699a1f56f"
  EXPECTED_MULTIPLY_ADDS  = 805_306_368_i64

  record Snapshot,
    total_bytes : Int64,
    live_managed_bytes : Int64,
    rss_kb : Int64

  def snapshot : Snapshot
    GC.collect
    stats = GC.stats
    Snapshot.new(
      stats.total_bytes.to_i64,
      stats.heap_size.to_i64 - stats.free_bytes.to_i64 - stats.unmapped_bytes.to_i64,
      current_rss_kb
    )
  end

  def current_rss_kb : Int64
    output = IO::Memory.new
    status = Process.run(
      "ps",
      ["-o", "rss=", "-p", Process.pid.to_s],
      output: output,
      error: Process::Redirect::Close
    )
    raise "RSS sensor command failed" unless status.success?
    output.to_s.strip.to_i64? || raise "RSS sensor returned no integer value"
  rescue ex : IO::Error
    raise "RSS sensor unavailable: #{ex.message}"
  end

  def finite_tensor?(tensor : ML::Tensor) : Bool
    values = tensor.cpu_read
    index = 0
    while index < values.size
      return false unless values[index].finite?
      index += 1
    end
    true
  end

  def fixture_path(relative : String) : String
    File.expand_path(File.join(__DIR__, "../../fixtures", relative))
  end
end

describe "DINOv3 real model-scale embedding forward" do
  it "runs one explicitly budgeted 512px CPU projection" do
    checkpoint = ENV["DINO_V3_CHECKPOINT"]? ||
                 raise "DINO_V3_CHECKPOINT must name the user-local model.safetensors"
    max_parameter_bytes = (
      ENV["DINO_V3_MAX_PARAMETER_BYTES"]? ||
      ML::Vision::DinoV3::DinoV3EmbeddingCaller::MAX_PARAMETER_BYTES.to_s
    ).to_i64
    rss_limit_mb = CogniSpecRSSGuard.limit_mb?
    raise "COGNI_SPEC_MAX_RSS_MB must be set for the model-scale probe" if rss_limit_mb <= 0
    max_multiply_adds = ENV["DINO_V3_FORWARD_MAX_MULTIPLY_ADDS"]?.try(&.to_i64?) ||
                        raise "DINO_V3_FORWARD_MAX_MULTIPLY_ADDS must be set explicitly"
    unless max_multiply_adds == DinoV3ModelScaleForwardProbe::EXPECTED_MULTIPLY_ADDS
      raise "DINO_V3_FORWARD_MAX_MULTIPLY_ADDS must equal the pinned 512px hidden-1024 projection cost"
    end

    certificate = ML::Vision::DinoV3::ConfigCertificate.parse(
      File.read(DinoV3ModelScaleForwardProbe.fixture_path("trellis2/dino_v3_config_certificate_v1.json")),
      source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
      source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
    )
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      File.read(DinoV3ModelScaleForwardProbe.fixture_path("trellis2/dino_v3_checkpoint_manifest_v1.json")),
      certificate: certificate
    )
    inventory = ML::Vision::DinoV3::CheckpointInventory.load(checkpoint, manifest: manifest)
    semantic = ML::Vision::DinoV3::SemanticInventory.bind(
      inventory,
      certificate: certificate,
      runtime: runtime
    )
    receipt = ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
      inventory,
      manifest: manifest
    )

    default_budget_rejected = false
    begin
      ML::Vision::DinoV3::DinoV3EmbeddingCaller.preflight(
        certificate,
        runtime,
        DinoV3ModelScaleForwardProbe::PROBE_EDGE
      )
    rescue ML::Vision::DinoV3::EmbeddingBudgetError
      default_budget_rejected = true
    end
    raise "production embedding budget did not reject the model-scale plan" unless default_budget_rejected

    before_load = DinoV3ModelScaleForwardProbe.snapshot
    caller = ML::Vision::DinoV3::DinoV3EmbeddingCaller.from_checkpoint(
      inventory,
      certificate: certificate,
      runtime: runtime,
      semantic: semantic,
      digest: receipt,
      max_parameter_bytes: max_parameter_bytes
    )
    after_load = DinoV3ModelScaleForwardProbe.snapshot

    plan = ML::Vision::DinoV3::EmbeddingCPU.preflight_model_scale_probe(
      caller.config,
      DinoV3ModelScaleForwardProbe::PROBE_EDGE,
      max_multiply_adds: max_multiply_adds
    )
    input = ML::Tensor.new(
      ML::Shape.new(1_i32, 3_i32, DinoV3ModelScaleForwardProbe::PROBE_EDGE,
        DinoV3ModelScaleForwardProbe::PROBE_EDGE),
      device: ML::Tensor::Device::CPU
    )
    before_forward = DinoV3ModelScaleForwardProbe.snapshot
    result = caller.forward_for_model_scale_probe(
      input,
      max_multiply_adds: max_multiply_adds
    )
    after_forward = DinoV3ModelScaleForwardProbe.snapshot

    result.patches.shape.to_a.should eq([1_i32, 1024_i32, 1024_i32])
    result.embeddings.shape.to_a.should eq([1_i32, 1029_i32, 1024_i32])
    result.patch_coordinates.shape.to_a.should eq([1024_i32, 2_i32])
    result.rope_cos.shape.to_a.should eq([1024_i32, 64_i32])
    result.rope_sin.shape.to_a.should eq([1024_i32, 64_i32])
    result.embeddings.numel.should eq(1_053_696)
    finite_output = DinoV3ModelScaleForwardProbe.finite_tensor?(result.embeddings)
    finite_output.should be_true

    file_bytes = inventory.file_byte_length
    parameter_bytes = caller.parameters.parameter_byte_length
    parameter_sha256 = caller.parameters.f32le_sha256
    file_bytes.should eq(DinoV3ModelScaleForwardProbe::EXPECTED_FILE_BYTES)
    parameter_bytes.should eq(DinoV3ModelScaleForwardProbe::EXPECTED_PARAMETER_BYTES)
    parameter_sha256.should eq(DinoV3ModelScaleForwardProbe::EXPECTED_PARAMETER_SHA256)
    plan.output_bytes.should eq(8_941_568_i64)

    load_allocation_bytes = after_load.total_bytes - before_load.total_bytes
    forward_allocation_bytes = after_forward.total_bytes - before_forward.total_bytes
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "dinov3-real-model-scale-forward-v1"
        json.field "scope", "darwin-local-manual"
        json.field "mode", "real-embedding-forward"
        json.field "input_edge", DinoV3ModelScaleForwardProbe::PROBE_EDGE
        json.field "checkpoint_bytes", file_bytes
        json.field "parameter_bytes", parameter_bytes
        json.field "parameter_f32le_sha256", parameter_sha256
        json.field "multiply_adds", plan.multiply_adds
        json.field "max_multiply_adds", max_multiply_adds
        json.field "output_bytes", plan.output_bytes
        json.field "output_finite", finite_output
        json.field "default_budget_rejected", default_budget_rejected
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "live_managed_kind", "post-gc-managed-heap-orientation"
        json.field "rss_kind", "process-resident-orientation"
        json.field "load_allocation_bytes", load_allocation_bytes
        json.field "load_held_live_managed_delta",
          after_load.live_managed_bytes - before_load.live_managed_bytes
        json.field "load_held_rss_delta_kb", after_load.rss_kb - before_load.rss_kb
        json.field "forward_allocation_bytes", forward_allocation_bytes
        json.field "forward_held_live_managed_delta",
          after_forward.live_managed_bytes - before_forward.live_managed_bytes
        json.field "forward_held_rss_delta_kb", after_forward.rss_kb - before_forward.rss_kb
        json.field "rss_guard_limit_mb", rss_limit_mb
        json.field "forward", "cpu-reference-ran"
        json.field "blocks_loaded", 0
        json.field "native_peak_memory", "not-measured"
        json.field "true_peak_rss", "not-measured"
        json.field "metal", "not-run"
      end
    end
    puts measurement
  end
end
{% end %}
