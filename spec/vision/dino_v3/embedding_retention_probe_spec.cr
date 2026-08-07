# Opt-in fresh-process Darwin CPU measurement for materializing the four real
# DINOv3 embedding roles. Run after the separate 16 MiB RSS sensor control.
# This probe does not execute an embedding forward, transformer blocks, Metal,
# or GPU code. GC/RSS values are local orientation signals, not native or peak
# memory certificates.

{% if flag?(:dinov3_embedding_retention_probe) %}
{% unless flag?(:darwin) %}
  {% raise "DINOv3 embedding retention probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "digest/sha256"
require "../../spec_helper"
require "../../../src/ml/vision/dino_v3"

module DinoV3EmbeddingRetentionProbe
  extend self

  EXPECTED_FILE_BYTES       = 1_212_559_808_i64
  EXPECTED_PARAMETER_BYTES  = 3_170_304_i64
  EXPECTED_PARAMETER_SHA256 = "8124dfb9859ecbb6d33908bcd013e32a469897140cb413c2aa5b629699a1f56f"

  record Snapshot,
    total_bytes : Int64,
    live_managed_bytes : Int64,
    rss_kb : Int64

  record Materialization,
    file_bytes : Int64,
    parameter_bytes : Int64,
    parameter_sha256 : String,
    held : Snapshot

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

  def materialize(path : String, max_parameter_bytes : Int64) : Materialization
    certificate = ML::Vision::DinoV3::ConfigCertificate.parse(
      File.read(fixture_path("trellis2/dino_v3_config_certificate_v1.json")),
      source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
      source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
    )
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      File.read(fixture_path("trellis2/dino_v3_checkpoint_manifest_v1.json")),
      certificate: certificate
    )
    inventory = ML::Vision::DinoV3::CheckpointInventory.load(path, manifest: manifest)
    semantic = ML::Vision::DinoV3::SemanticInventory.bind(
      inventory,
      certificate: certificate,
      runtime: runtime
    )
    receipt = ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
      inventory,
      manifest: manifest
    )

    caller : ML::Vision::DinoV3::DinoV3EmbeddingCaller? =
      ML::Vision::DinoV3::DinoV3EmbeddingCaller.from_checkpoint(
        inventory,
        certificate: certificate,
        runtime: runtime,
        semantic: semantic,
        digest: receipt,
        max_parameter_bytes: max_parameter_bytes
      )
    loaded = caller.not_nil!
    parameter_bytes = loaded.parameters.parameter_byte_length
    parameter_sha256 = loaded.parameters.f32le_sha256
    held = snapshot

    Materialization.new(
      inventory.file_byte_length,
      parameter_bytes,
      parameter_sha256,
      held
    )
  end

  private def fixture_path(relative : String) : String
    File.expand_path(File.join(__DIR__, "../../fixtures", relative))
  end
end

describe "DINOv3 real embedding-role retention" do
  it "measures bounded role materialization without running the encoder" do
    path = ENV["DINO_V3_CHECKPOINT"]? ||
           raise "DINO_V3_CHECKPOINT must name the user-local model.safetensors"
    max_parameter_bytes = (
      ENV["DINO_V3_MAX_PARAMETER_BYTES"]? ||
      ML::Vision::DinoV3::DinoV3EmbeddingCaller::MAX_PARAMETER_BYTES.to_s
    ).to_i64

    before = DinoV3EmbeddingRetentionProbe.snapshot
    loaded = DinoV3EmbeddingRetentionProbe.materialize(path, max_parameter_bytes)
    dropped = DinoV3EmbeddingRetentionProbe.snapshot

    loaded.file_bytes.should eq(DinoV3EmbeddingRetentionProbe::EXPECTED_FILE_BYTES)
    loaded.parameter_bytes.should eq(DinoV3EmbeddingRetentionProbe::EXPECTED_PARAMETER_BYTES)
    loaded.parameter_sha256.should eq(
      DinoV3EmbeddingRetentionProbe::EXPECTED_PARAMETER_SHA256
    )

    allocation_bytes = loaded.held.total_bytes - before.total_bytes
    held_live_delta = loaded.held.live_managed_bytes - before.live_managed_bytes
    released_live = loaded.held.live_managed_bytes - dropped.live_managed_bytes
    allocation_bytes.should be >= loaded.parameter_bytes
    held_live_delta.should be > 0
    released_live.should be > 0

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "dinov3-real-embedding-role-retention-v1"
        json.field "scope", "darwin-local-manual"
        json.field "mode", "real-embedding-roles"
        json.field "checkpoint_bytes", loaded.file_bytes
        json.field "parameter_bytes", loaded.parameter_bytes
        json.field "parameter_f32le_sha256", loaded.parameter_sha256
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "live_managed_kind", "post-gc-managed-heap-orientation"
        json.field "rss_kind", "process-resident-orientation"
        json.field "allocation_bytes", allocation_bytes
        json.field "held_live_managed_delta", held_live_delta
        json.field "held_rss_delta_kb", loaded.held.rss_kb - before.rss_kb
        json.field "dropped_live_managed_delta",
          dropped.live_managed_bytes - before.live_managed_bytes
        json.field "dropped_rss_delta_kb", dropped.rss_kb - before.rss_kb
        json.field "released_live_managed_bytes", released_live
        json.field "forward", "not-run"
        json.field "blocks_loaded", 0
        json.field "native_peak_memory", "not-measured"
        json.field "metal", "not-run"
      end
    end
    puts measurement
  end
end
{% end %}
