require "digest/sha256"
require "json"
require "./support/hy_nomas_cache_qbit"
require "./support/hy_nomas_cache_qbit_checkpoint"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasCacheQBitCheckpointProbe
  extend self

  ROW_VALUES = HyNomasCacheQBit::ROW_VALUES
  ROWS       = 6
  COUNT      = ROW_VALUES * ROWS
  NAME       = "history.layer0.key"
  CONTRACT   = "hy-nomas-cache-qbit-checkpoint-probe-v1"

  class RetainedSaveResources
    def initialize(@control_plan : HyNomasCacheQBit::Plan,
                   @pack_plan : HyNomasCacheQBitCheckpoint::PackPlan,
                   @control : ML::MetalBuffer,
                   @packed : ML::MetalBuffer,
                   @timeline : ML::MetalBuffer,
                   @chain : ML::Metal::MPSCommandChain)
    end
  end

  class RetainedRestoreResources
    def initialize(@restore_plan : HyNomasCacheQBitCheckpoint::RestorePlan,
                   @cache : ML::MetalBuffer,
                   @timeline : ML::MetalBuffer,
                   @chain : ML::Metal::MPSCommandChain)
    end
  end

  @@unreleased_save = [] of RetainedSaveResources
  @@unreleased_restore = [] of RetainedRestoreResources

  def source_values : Array(Float32)
    values = Array(Float32).new(COUNT, 0.0_f32)

    # Non-binary constant exercises exact zero-sigma headers.
    ROW_VALUES.times { |index| values[index] = 0.1_f32 }

    # Symmetric and approximately Gaussian rows exercise bitplanes.
    ROW_VALUES.times do |index|
      values[ROW_VALUES + index] = index.even? ? 1.0_f32 : -1.0_f32
    end
    state = 8042_u32
    ROW_VALUES.times do |index|
      total = 0.0_f32
      12.times do
        state = ((state.to_u64 &* 1_664_525_u64 &+ 1_013_904_223_u64) & 0xffff_ffff_u64).to_u32
        unit = ((state >> 8) & 0x00ff_ffff_u32).to_f32 / 16_777_216.0_f32
        total += unit - 0.5_f32
      end
      values[2 * ROW_VALUES + index] = total
    end

    # Sparse tail values exercise p8; a finite high outlier must use exact F32.
    8.times { |index| values[3 * ROW_VALUES + index] = 1.0_f32 }
    values[4 * ROW_VALUES] = 20.0_f32
    # Constant -0 must retain its sign bit through the zero-sigma decoder.
    ROW_VALUES.times { |index| values[5 * ROW_VALUES + index] = -0.0_f32 }
    values
  end

  def timeline_values : Array(Float32)
    Array(Float32).new(17) { |index| (index.to_f32 * 0.25_f32) - 1.5_f32 }
  end

  def float_bytes(values : Array(Float32)) : Bytes
    bytes = Bytes.new(values.size * sizeof(Float32))
    values.each_with_index do |value, index|
      HyNomasCacheQBitCheckpoint.write_f32(bytes, index * sizeof(Float32), value)
    end
    bytes
  end

  def read_float_bytes(bytes : Bytes) : Array(Float32)
    unless bytes.size % sizeof(Float32) == 0
      raise ArgumentError.new("checkpoint probe reference has a partial F32 value")
    end
    Array(Float32).new(bytes.size // sizeof(Float32)) do |index|
      HyNomasCacheQBitCheckpoint.read_f32(bytes, index * sizeof(Float32))
    end
  end

  def write_reference(path : String, values : Array(Float32)) : String
    raise ArgumentError.new("checkpoint probe reference path already exists") if File.info?(path, false)
    bytes = float_bytes(values)
    File.open(path, "wb") { |io| io.write(bytes) }
    Digest::SHA256.hexdigest(bytes)
  end

  def read_reference(path : String) : {Array(Float32), String}
    size = File.info(path, false).size
    bytes = Bytes.new(size.to_i)
    File.open(path, "rb") { |io| io.read_fully(bytes) }
    {read_float_bytes(bytes), Digest::SHA256.hexdigest(bytes)}
  end

  def assert_bits_equal!(expected : Array(Float32), actual : Array(Float32), label : String) : Nil
    unless expected.size == actual.size
      raise "#{label} size mismatch: expected #{expected.size}, got #{actual.size}"
    end
    expected.each_with_index do |value, index|
      unless value.unsafe_as(UInt32) == actual[index].unsafe_as(UInt32)
        raise "#{label} changed F32 bits at index #{index}"
      end
    end
  end

  def assert_no_encode_effects!(allocations_before : Int64, readbacks_before : Int64,
                                chain : ML::Metal::MPSCommandChain) : {Int64, Int64}
    allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
    unless allocations == 0 && readbacks == 0 && chain.wait_count == 0
      raise "checkpoint encode performed named allocations, readbacks, or waits"
    end
    {allocations, readbacks}
  end

  def cleanup_save!(control_plan : HyNomasCacheQBit::Plan,
                    pack_plan : HyNomasCacheQBitCheckpoint::PackPlan,
                    control : ML::MetalBuffer, packed : ML::MetalBuffer,
                    timeline : ML::MetalBuffer,
                    chain : ML::Metal::MPSCommandChain) : Nil
    unless chain.completed?
      begin
        chain.commit_and_wait
      rescue
      end
    end
    if chain.completed?
      control_plan.release_after_drain(chain)
      pack_plan.release_after_drain(chain)
      control.release
      packed.release
      timeline.release
      chain.release
    else
      @@unreleased_save << RetainedSaveResources.new(control_plan, pack_plan,
        control, packed, timeline, chain)
    end
  end

  def cleanup_restore!(restore_plan : HyNomasCacheQBitCheckpoint::RestorePlan?,
                       cache : ML::MetalBuffer?, timeline : ML::MetalBuffer?,
                       chain : ML::Metal::MPSCommandChain?) : Nil
    unless active_chain = chain
      restore_plan.try(&.release_after_drain(nil))
      cache.try(&.release)
      timeline.try(&.release)
      return
    end
    unless active_chain.completed?
      begin
        active_chain.commit_and_wait
      rescue
      end
    end
    if active_chain.completed?
      if plan = restore_plan
        plan.release_after_drain(active_chain)
      end
      cache.try(&.release)
      timeline.try(&.release)
      active_chain.release
    elsif plan = restore_plan
      if cache && timeline
        @@unreleased_restore << RetainedRestoreResources.new(plan,
          cache.not_nil!, timeline.not_nil!, active_chain)
      end
    end
  end

  def save(checkpoint_path : String, reference_path : String) : Nil
    raise "Metal unavailable" unless ML::Metal::Device.init!
    values = source_values
    timeline = timeline_values
    control = ML::MetalBuffer.from_array(values)
    packed = ML::MetalBuffer.from_array(values)
    timeline_buffer = ML::MetalBuffer.from_array(timeline)
    binding = HyNomasCacheQBit::Binding.new(NAME, packed, COUNT.to_i32)
    timeline_binding = HyNomasCacheQBit::Binding.new("timeline", timeline_buffer, timeline.size.to_i32)
    control_binding = HyNomasCacheQBit::Binding.new(NAME, control, COUNT.to_i32)
    control_plan = HyNomasCacheQBit::Plan.new([control_binding])
    pack_plan = HyNomasCacheQBitCheckpoint::PackPlan.new([binding])
    chain = ML::Metal::MPSCommandChain.new
    encode_allocations = 0_i64
    encode_readbacks = 0_i64

    begin
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      control_plan.encode(chain)
      pack_plan.encode_pack(chain)
      encode_allocations, encode_readbacks = assert_no_encode_effects!(
        allocations_before, readbacks_before, chain
      )

      chain.commit_and_wait
      raise "checkpoint save process must have exactly one terminal wait" unless chain.wait_count == 1
      control_report = control_plan.report_after_drain(chain)
      HyNomasCacheQBit::Plan.validate_report!(control_report)
      packed_report = pack_plan.report_after_drain(chain)
      unless packed_report["invalid_rows"].as_i64 == 0
        raise "checkpoint pack reported invalid rows"
      end
      packed_tiers = packed_report["tier_counts"]
      expected_tiers = {"p4" => 3_i64, "p5" => 1_i64, "p8" => 1_i64, "f32" => 1_i64}
      expected_tiers.each do |tier, expected|
        unless packed_tiers[tier].as_i64 == expected
          raise "checkpoint fixture expected #{expected} #{tier} rows, got #{packed_tiers[tier].as_i64}"
        end
      end

      reference = control.read(COUNT.to_i32)
      reference_sha = write_reference(reference_path, reference)
      bundle_sha = pack_plan.write_after_drain(chain, checkpoint_path, CONTRACT, timeline_binding)
      final_report = pack_plan.report_after_drain(chain)
      unless final_report["disk_bytes_known"].as_bool &&
             final_report["logical_packed_kv_bytes"].as_i64 == final_report["logical_packed_bytes"].as_i64
        raise "checkpoint report did not contain final compact byte counts"
      end

      puts(JSON.build do |json|
        json.object do
          json.field "schema", "hy-nomas-cache-qbit-checkpoint-probe-v1"
          json.field "mode", "save"
          json.field "bundle_sha256", bundle_sha
          json.field "reference_sha256", reference_sha
          json.field "reference_values", reference.size
          json.field "report", final_report
          json.field "encode_named_allocations", encode_allocations
          json.field "encode_readbacks", encode_readbacks
          json.field "terminal_waits", chain.wait_count
        end
      end)
    ensure
      cleanup_save!(control_plan, pack_plan, control, packed, timeline_buffer, chain)
    end
  end

  def restore(checkpoint_path : String, bundle_sha : String,
              reference_path : String) : Nil
    # Authenticate and preflight before allocating the cache/timeline Metal
    # buffers that stand in for the model's live destination storage.
    timeline = timeline_values
    logical_inventory = {
      "timeline" => timeline.size.to_i64 * sizeof(Float32),
      NAME       => COUNT.to_i64 * sizeof(Float32),
    }
    preflight = HyNomasCacheQBitCheckpoint::Preflight.open(
      checkpoint_path, bundle_sha, CONTRACT, logical_inventory
    )
    cache = nil.as(ML::MetalBuffer?)
    timeline_buffer = nil.as(ML::MetalBuffer?)
    restore_plan = nil.as(HyNomasCacheQBitCheckpoint::RestorePlan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)

    begin
      raise "Metal unavailable" unless ML::Metal::Device.init!
      cache = ML::MetalBuffer.from_array(source_values)
      timeline_buffer = ML::MetalBuffer.from_array(Array(Float32).new(timeline.size, 99.0_f32))
      binding = HyNomasCacheQBit::Binding.new(NAME, cache.not_nil!, COUNT.to_i32)
      timeline_binding = HyNomasCacheQBit::Binding.new("timeline", timeline_buffer.not_nil!, timeline.size.to_i32)
      restore_plan = HyNomasCacheQBitCheckpoint::RestorePlan.new([binding], preflight, timeline_binding)
      preflight.close

      active_chain = ML::Metal::MPSCommandChain.new
      chain = active_chain
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      restore_plan.not_nil!.encode_restore(active_chain)
      encode_allocations, encode_readbacks = assert_no_encode_effects!(
        allocations_before, readbacks_before, active_chain
      )
      active_chain.commit_and_wait
      raise "checkpoint restore process must have exactly one terminal wait" unless active_chain.wait_count == 1

      expected_cache, reference_sha = read_reference(reference_path)
      actual_cache = cache.not_nil!.read(COUNT.to_i32)
      raise "restored cache contains nonfinite values" unless actual_cache.all?(&.finite?)
      assert_bits_equal!(expected_cache, actual_cache, "restored cache")
      ROW_VALUES.times do |index|
        unless actual_cache[index].unsafe_as(UInt32) == 0.1_f32.unsafe_as(UInt32)
          raise "non-binary constant row was not lossless"
        end
        unless actual_cache[5 * ROW_VALUES + index].unsafe_as(UInt32) == (-0.0_f32).unsafe_as(UInt32)
          raise "negative-zero constant row lost its sign bit"
        end
      end
      actual_timeline = timeline_buffer.not_nil!.read(timeline.size.to_i32)
      assert_bits_equal!(timeline, actual_timeline, "restored timeline")

      puts(JSON.build do |json|
        json.object do
          json.field "schema", "hy-nomas-cache-qbit-checkpoint-probe-v1"
          json.field "mode", "restore"
          json.field "bundle_sha256", bundle_sha
          json.field "reference_sha256", reference_sha
          json.field "cache_values_equal_bitwise", true
          json.field "timeline_equal_bitwise", true
          json.field "encode_named_allocations", encode_allocations
          json.field "encode_readbacks", encode_readbacks
          json.field "terminal_waits", active_chain.wait_count
        end
      end)
    ensure
      preflight.close
      cleanup_restore!(restore_plan, cache, timeline_buffer, chain)
    end
  end
end

if ARGV.size == 3 && ARGV[0] == "save"
  HyNomasCacheQBitCheckpointProbe.save(ARGV[1], ARGV[2])
elsif ARGV.size == 4 && ARGV[0] == "restore"
  HyNomasCacheQBitCheckpointProbe.restore(ARGV[1], ARGV[2], ARGV[3])
else
  raise "usage: hy_nomas_cache_qbit_checkpoint_probe save PATH REF_PATH | restore PATH SHA256 REF_PATH"
end
