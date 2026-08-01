# Opt-in fresh-process Darwin CPU retention/allocation probes.
#
# These probes emit GC/RSS observations; they do not establish allocator,
# device-memory, or Metal fitness. The ps-based RSS sensor and its threshold
# are a local manual qualification, not a portable CI contract. Run each
# define in a separate process. The allocation-attribution mode does not call
# ps and reports cumulative allocation traffic only.

{% if (flag?(:trellis2_retention_sensor_probe) && flag?(:trellis2_retention_cache_probe)) ||
        (flag?(:trellis2_retention_sensor_probe) && flag?(:trellis2_allocation_attribution_probe)) ||
        (flag?(:trellis2_retention_cache_probe) && flag?(:trellis2_allocation_attribution_probe)) %}
  {% raise "retention and allocation probes must run in separate fresh processes" %}
{% end %}

{% if flag?(:trellis2_retention_sensor_probe) || flag?(:trellis2_retention_cache_probe) ||
        flag?(:trellis2_allocation_attribution_probe) %}
{% unless flag?(:darwin) %}
  {% raise "retention/allocation probes are admitted only for local Darwin measurement" %}
{% end %}

require "../../spec_helper"
require "json"
require "../../../src/ml/three_d/trellis2/cache_adapter"

module Trellis2RetentionProbe
  extend self

  SENSOR_BYTES            = 16_i64 * 1024_i64 * 1024_i64
  SENSOR_MIN_RSS_DELTA_KB =  8_i64 * 1024_i64
  CACHE_HITS_PER_WINDOW   = 20_000

  record Snapshot,
    heap_size : Int64,
    total_bytes : Int64,
    free_bytes : Int64,
    unmapped_bytes : Int64,
    rss_kb : Int64 do
    def gc_heap_minus_free_unmapped : Int64
      heap_size - free_bytes - unmapped_bytes
    end
  end

  def snapshot : Snapshot
    GC.collect
    stats = GC.stats
    Snapshot.new(
      stats.heap_size.to_i64,
      stats.total_bytes.to_i64,
      stats.free_bytes.to_i64,
      stats.unmapped_bytes.to_i64,
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

  def gc_total_bytes : Int64
    GC.collect
    GC.stats.total_bytes.to_i64
  end

  def emit_snapshot(builder : JSON::Builder, snapshot : Snapshot) : Nil
    builder.object do
      builder.field "gc_heap_size", snapshot.heap_size
      builder.field "gc_total_bytes", snapshot.total_bytes
      builder.field "gc_free_bytes", snapshot.free_bytes
      builder.field "gc_unmapped_bytes", snapshot.unmapped_bytes
      builder.field "gc_heap_minus_free_unmapped", snapshot.gc_heap_minus_free_unmapped
      builder.field "rss_kb", snapshot.rss_kb
    end
  end

  def emit_delta(builder : JSON::Builder, before : Snapshot, after : Snapshot) : Nil
    builder.object do
      builder.field "gc_heap_size", after.heap_size - before.heap_size
      builder.field "gc_total_bytes", after.total_bytes - before.total_bytes
      builder.field "gc_heap_minus_free_unmapped",
        after.gc_heap_minus_free_unmapped - before.gc_heap_minus_free_unmapped
      builder.field "rss_kb", after.rss_kb - before.rss_kb
    end
  end

  def cache_contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract
    profile = ML::ThreeD::Trellis2::DenseStageResourceProfile.new(
      id: "dense-cache-retention",
      in_channels: 4,
      model_channels: 16,
      context_channels: 12,
      out_channels: 6,
      num_heads: 2,
      mlp_hidden_channels: 32,
      frequency_dim: 8,
      batch_buckets: [1_i32],
      voxel_buckets: [8_i32],
      context_buckets: [2_i32]
    )
    abi = ML::ThreeD::Trellis2::DenseKernelABI.new(
      source_digest: "3" * 64,
      device_family: "cpu-oracle",
      compiler_abi: "t2n2d1b-v1",
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
      kernel_variants: ["mlp"],
      dtypes: [ML::DType::F32],
      cache_owner: ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU::CACHE_OWNER,
      kernel_abi: abi,
      max_axis_padding_ratio: 1.0,
      max_single_tensor_bytes: 1_i64 * 1024_i64 * 1024_i64,
      max_declared_activation_bytes: 8_i64 * 1024_i64 * 1024_i64
    )
  end

  def cache_key(
    contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract,
  ) : ML::ThreeD::Trellis2::KernelSpecializationKey
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "dense-cache-retention",
      1,
      8,
      2,
      ML::DType::F32
    )
    contract.plan(request).kernel_keys.first
  end
end

{% if flag?(:trellis2_retention_sensor_probe) %}
describe "TRELLIS.2 retention sensor control" do
  it "detects a known retained sixteen MiB allocation" do
    before = Trellis2RetentionProbe.snapshot
    bytes = Array(UInt8).new(
      Trellis2RetentionProbe::SENSOR_BYTES.to_i,
      0xa5_u8
    )
    checksum = bytes.sum(0_i64) { |value| value.to_i64 }
    after = Trellis2RetentionProbe.snapshot
    total_delta = after.total_bytes - before.total_bytes
    rss_delta = after.rss_kb - before.rss_kb

    bytes.first.should eq(0xa5_u8)
    checksum.should eq(0xa5_i64 * Trellis2RetentionProbe::SENSOR_BYTES)
    total_delta.should be >= Trellis2RetentionProbe::SENSOR_BYTES
    rss_delta.should be >= Trellis2RetentionProbe::SENSOR_MIN_RSS_DELTA_KB

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-retention-sensor-v1"
        json.field "scope", "darwin-local-manual"
        json.field "rss_sensor", "ps-rss-kb"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "rss_kind", "process-resident-orientation"
        json.field "sensor_bytes", Trellis2RetentionProbe::SENSOR_BYTES
        json.field "before" do
          Trellis2RetentionProbe.emit_snapshot(json, before)
        end
        json.field "after" do
          Trellis2RetentionProbe.emit_snapshot(json, after)
        end
        json.field "delta" do
          Trellis2RetentionProbe.emit_delta(json, before, after)
        end
      end
    end
    puts measurement
  end
end
{% end %}

{% if flag?(:trellis2_retention_cache_probe) %}
describe "TRELLIS.2 local cache retention orientation" do
  it "keeps one compiled entry across two repeated-hit windows" do
    contract = Trellis2RetentionProbe.cache_contract
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
    compile_calls = [] of String
    compiler = ->(key : ML::ThreeD::Trellis2::KernelSpecializationKey) do
      compile_calls << key.canonical
      "compiled:#{key.canonical}"
    end
    adapter = ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU.new(
      ledger,
      1,
      compiler
    )
    key = Trellis2RetentionProbe.cache_key(contract)
    artifact = adapter.fetch!(key)
    before = Trellis2RetentionProbe.snapshot

    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      unless adapter.fetch!(key).same?(artifact)
        raise "cache hit artifact identity changed"
      end
    end
    after_first = Trellis2RetentionProbe.snapshot

    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      unless adapter.fetch!(key).same?(artifact)
        raise "cache hit artifact identity changed"
      end
    end
    after_second = Trellis2RetentionProbe.snapshot

    diagnostics = adapter.diagnostics
    diagnostics.lookups.should eq(2 * Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW + 1)
    diagnostics.hits.should eq(2 * Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW)
    diagnostics.misses.should eq(1)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.capacity_refusals.should eq(0)
    diagnostics.entries.should eq(1)
    compile_calls.should eq([key.canonical])

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-local-cache-retention-v1"
        json.field "scope", "darwin-local-manual"
        json.field "rss_sensor", "ps-rss-kb"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "rss_kind", "process-resident-orientation"
        json.field "hits_per_window", Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW
        json.field "before" do
          Trellis2RetentionProbe.emit_snapshot(json, before)
        end
        json.field "after_first" do
          Trellis2RetentionProbe.emit_snapshot(json, after_first)
        end
        json.field "after_second" do
          Trellis2RetentionProbe.emit_snapshot(json, after_second)
        end
        json.field "first_delta" do
          Trellis2RetentionProbe.emit_delta(json, before, after_first)
        end
        json.field "second_delta" do
          Trellis2RetentionProbe.emit_delta(json, after_first, after_second)
        end
        json.field "diagnostics" do
          json.object do
            json.field "lookups", diagnostics.lookups
            json.field "hits", diagnostics.hits
            json.field "misses", diagnostics.misses
            json.field "compile_attempts", diagnostics.compile_attempts
            json.field "capacity_refusals", diagnostics.capacity_refusals
            json.field "entries", diagnostics.entries
          end
        end
      end
    end
    puts measurement
  end
end
{% end %}

{% if flag?(:trellis2_allocation_attribution_probe) %}
describe "TRELLIS.2 local cache allocation attribution" do
  it "separates no-op, ledger admission, and adapter-hit traffic" do
    contract = Trellis2RetentionProbe.cache_contract
    ledger = ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(contract, 1)
    compile_calls = [] of String
    compiler = ->(key : ML::ThreeD::Trellis2::KernelSpecializationKey) do
      compile_calls << key.canonical
      "compiled:#{key.canonical}"
    end
    adapter = ML::ThreeD::Trellis2::BoundedKernelCacheAdapterCPU.new(
      ledger,
      1,
      compiler
    )
    key = Trellis2RetentionProbe.cache_key(contract)
    artifact = adapter.fetch!(key)

    noop_before = Trellis2RetentionProbe.gc_total_bytes
    noop_sink = 0_i64
    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      noop_sink += 1
    end
    noop_after = Trellis2RetentionProbe.gc_total_bytes

    ledger_before = Trellis2RetentionProbe.gc_total_bytes
    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      ledger.admit!([key])
    end
    ledger_after = Trellis2RetentionProbe.gc_total_bytes

    adapter_before = Trellis2RetentionProbe.gc_total_bytes
    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      unless adapter.fetch!(key).same?(artifact)
        raise "cache hit artifact identity changed"
      end
    end
    adapter_after = Trellis2RetentionProbe.gc_total_bytes

    adapter_reverse_before = Trellis2RetentionProbe.gc_total_bytes
    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      unless adapter.fetch!(key).same?(artifact)
        raise "cache hit artifact identity changed"
      end
    end
    adapter_reverse_after = Trellis2RetentionProbe.gc_total_bytes

    ledger_reverse_before = Trellis2RetentionProbe.gc_total_bytes
    Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW.times do
      ledger.admit!([key])
    end
    ledger_reverse_after = Trellis2RetentionProbe.gc_total_bytes

    noop_bytes = noop_after - noop_before
    ledger_bytes = ledger_after - ledger_before
    adapter_bytes = adapter_after - adapter_before
    adapter_reverse_bytes = adapter_reverse_after - adapter_reverse_before
    ledger_reverse_bytes = ledger_reverse_after - ledger_reverse_before
    diagnostics = adapter.diagnostics

    noop_sink.should eq(Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW)
    ledger.size.should eq(1)
    diagnostics.lookups.should eq(2 * Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW + 1)
    diagnostics.hits.should eq(2 * Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW)
    diagnostics.misses.should eq(1)
    diagnostics.compile_attempts.should eq(1)
    diagnostics.capacity_refusals.should eq(0)
    diagnostics.entries.should eq(1)
    compile_calls.should eq([key.canonical])
    ledger_bytes.should be > noop_bytes
    adapter_bytes.should be > noop_bytes
    adapter_reverse_bytes.should be > noop_bytes
    ledger_reverse_bytes.should be > noop_bytes

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-local-cache-allocation-attribution-v1"
        json.field "scope", "darwin-local-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "window_order", [
          "noop",
          "round_a_ledger_only",
          "round_a_adapter_hit",
          "round_b_adapter_hit",
          "round_b_ledger_only",
        ]
        json.field "iterations_per_window", Trellis2RetentionProbe::CACHE_HITS_PER_WINDOW
        json.field "noop_bytes", noop_bytes
        json.field "round_a_ledger_bytes", ledger_bytes
        json.field "round_a_adapter_bytes", adapter_bytes
        json.field "round_a_adapter_minus_ledger_bytes", adapter_bytes - ledger_bytes
        json.field "round_b_adapter_bytes", adapter_reverse_bytes
        json.field "round_b_ledger_bytes", ledger_reverse_bytes
        json.field "round_b_adapter_minus_ledger_bytes", adapter_reverse_bytes - ledger_reverse_bytes
        json.field "diagnostics" do
          json.object do
            json.field "lookups", diagnostics.lookups
            json.field "hits", diagnostics.hits
            json.field "misses", diagnostics.misses
            json.field "compile_attempts", diagnostics.compile_attempts
            json.field "capacity_refusals", diagnostics.capacity_refusals
            json.field "entries", diagnostics.entries
          end
        end
      end
    end
    puts measurement
  end
end
{% end %}
{% end %}
