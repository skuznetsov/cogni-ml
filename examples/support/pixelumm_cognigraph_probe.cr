require "json"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_ffn_weight_reuse_probe"

# CPU-side schedule, lifecycle and report helpers for the bounded PixelUMM
# CogniGraph measurement. Native preparation is setup-only; request selection
# happens only after the preceding request's sole terminal wait and release.
module PixelUMMCogniGraphProbe
  extend self

  alias B = PixelUMMFFNWeightReuseProbe
  SCHEMA              = "cogni_ml.pixelumm_cognigraph_probe/v1"
  REQUEST_COUNT       = 40
  TIMED_REQUEST_COUNT = 32
  PAIR_PATTERN        = [false, true, true, false, true, false, false, true]
  NATIVE_STAT_NAMES   = ["graphs", "ops", "waves", "barriers", "maxwidth",
                         "native_dispatches_perblock", "baseline_encoders_perblock",
                         "candidate_encoders_perblock_no_captures"]

  alias Trial = NamedTuple(index: Int32, enabled: Bool, kind: String,
    capture_mode: Bool, positive_nan_control: Bool, timing_eligible: Bool)
  alias NativeStats = NamedTuple(graphs: UInt64, ops: UInt64, waves: UInt64,
    barriers: UInt64, maxwidth: UInt64, native_dispatches_perblock: UInt64,
    baseline_encoders_perblock: UInt64,
    candidate_encoders_perblock_no_captures: UInt64)

  def schedule : Array(Trial)
    entries = [] of Trial
    correctness_modes = [false, true, false, true]
    correctness_capture = [true, true, false, false]
    4.times do |index|
      entries << {index: index.to_i32, enabled: correctness_modes[index],
                  kind: "correctness", capture_mode: correctness_capture[index],
                  positive_nan_control: index == 0, timing_eligible: false}
    end

    [false, true, true, false].each_with_index do |enabled, offset|
      index = 4 + offset
      entries << {index: index.to_i32, enabled: enabled, kind: "warmup",
                  capture_mode: false, positive_nan_control: false,
                  timing_eligible: false}
    end

    TIMED_REQUEST_COUNT.times do |offset|
      index = 8 + offset
      entries << {index: index.to_i32,
                  enabled: PAIR_PATTERN[offset % PAIR_PATTERN.size],
                  kind: "timed", capture_mode: false,
                  positive_nan_control: false, timing_eligible: true}
    end
    validate_schedule!(entries)
    entries
  end

  def owner_entry(trial : Trial) : B::ScheduleEntry
    {index: trial[:index], mode: "baseline", kind: trial[:kind],
     capture_mode: trial[:capture_mode], baseline_schedule: "window-major",
     candidate_schedule: "block-major", timing_eligible: trial[:timing_eligible]}
  end

  def validate_schedule!(entries : Array(Trial)) : Nil
    unless entries.size == REQUEST_COUNT &&
           entries.map(&.[:index]) == (0...REQUEST_COUNT).map(&.to_i32)
      raise ArgumentError.new("CogniGraph probe requires 40 contiguous requests")
    end
    unless entries[0...4].map(&.[:enabled]) == [false, true, false, true] &&
           entries[0...4].map(&.[:capture_mode]) == [true, true, false, false] &&
           entries[0...4].all? { |entry| entry[:kind] == "correctness" && !entry[:timing_eligible] } &&
           entries[0...4].map(&.[:positive_nan_control]) == [true, false, false, false]
      raise ArgumentError.new("CogniGraph correctness/capture boundaries changed")
    end
    unless entries[4...8].map(&.[:enabled]) == [false, true, true, false] &&
           entries[4...8].all? { |entry| entry[:kind] == "warmup" && !entry[:capture_mode] &&
           !entry[:positive_nan_control] && !entry[:timing_eligible] }
      raise ArgumentError.new("CogniGraph ABBA warmup boundary changed")
    end
    timed = entries[8..]
    unless timed.size == TIMED_REQUEST_COUNT &&
           timed.each_slice(PAIR_PATTERN.size).all? { |chunk| chunk.map(&.[:enabled]) == PAIR_PATTERN } &&
           timed.count(&.[:enabled]) == TIMED_REQUEST_COUNT // 2 &&
           timed.all? { |entry| entry[:kind] == "timed" && !entry[:capture_mode] &&
           !entry[:positive_nan_control] && entry[:timing_eligible] }
      raise ArgumentError.new("CogniGraph timed ABBA/BAAB schedule is not balanced")
    end
  end

  # Enforces the external state transition contract. The caller performs the
  # native setter only when prepare_request! returns true, and does that before
  # begin_request!; finish requires an already completed one-wait request.
  class NativeIslandMode
    getter next_index : Int32 = 0
    getter enabled : Bool = false
    @known = false
    @prepared : NamedTuple(index: Int32, enabled: Bool)? = nil
    @active_index : Int32? = nil
    @last_wait_count = 0

    def prepare_request!(index : Int32, enabled : Bool) : Bool
      unless @active_index.nil? && @prepared.nil? && index == @next_index
        raise ArgumentError.new("CogniGraph mode can change only for the next idle request")
      end
      if @next_index > 0 && enabled != @enabled && @last_wait_count != 1
        raise ArgumentError.new("CogniGraph mode change requires the prior sole terminal wait")
      end
      changed = !@known || @enabled != enabled
      @enabled = enabled
      @known = true
      @prepared = {index: index, enabled: enabled}
      changed
    end

    def begin_request!(index : Int32, enabled : Bool) : Nil
      prepared = @prepared || raise ArgumentError.new("CogniGraph request mode was not prepared")
      unless prepared[:index] == index && prepared[:enabled] == enabled &&
             index == @next_index && @active_index.nil?
        raise ArgumentError.new("CogniGraph request does not match its prepared native mode")
      end
      @active_index = index
      @prepared = nil
    end

    def finish_request!(index : Int32, wait_count : Int32) : Nil
      unless @active_index == index && index == @next_index && wait_count == 1
        raise ArgumentError.new("CogniGraph request must finish after exactly one terminal wait")
      end
      @active_index = nil
      @last_wait_count = wait_count
      @next_index += 1
    end
  end

  lib PixelUMMCogniGraphHostClock
    fun clock_ns = clock_gettime_nsec_np(clock_id : UInt32) : UInt64
  end

  record HostPoint, wall_ns : UInt64, thread_cpu_ns : UInt64 do
    def self.now : self
      wall = PixelUMMCogniGraphHostClock.clock_ns(6_u32)    # CLOCK_MONOTONIC
      thread = PixelUMMCogniGraphHostClock.clock_ns(16_u32) # CLOCK_THREAD_CPUTIME_ID
      unless wall > 0 && thread > 0
        raise ArgumentError.new("invalid monotonic or thread CPU clock sample")
      end
      new(wall, thread)
    end
  end

  def host_thread_seconds(first : HostPoint, last : HostPoint) : Float64
    unless last.thread_cpu_ns >= first.thread_cpu_ns
      raise ArgumentError.new("thread CPU clock moved backwards")
    end
    (last.thread_cpu_ns - first.thread_cpu_ns).to_f64 / 1e9
  end

  def owner_handle_pointer(handle : UInt64) : Pointer(Void)
    raise ArgumentError.new("null native PixelUMM plan handle") if handle == 0_u64
    Pointer(Void).new(handle)
  end

  private def ffi_error_text(bytes : Bytes) : String
    length = bytes.index(0_u8) || bytes.size
    String.new(bytes.to_unsafe, length)
  end

  {% unless flag?(:cpu_only) %}
    def native_prepare_status(handle : UInt64) : Int32
      error = Bytes.new(1024, 0_u8)
      PixelUMMDecoderBlockMPSFFI.prepare_native_islands(
        owner_handle_pointer(handle), ML::Metal::CogniGraphSchedule.callback,
        error.to_unsafe, error.size)
    end

    def native_set_status(handle : UInt64, enabled : Int32) : Int32
      error = Bytes.new(1024, 0_u8)
      PixelUMMDecoderBlockMPSFFI.set_native_islands(
        owner_handle_pointer(handle), enabled, error.to_unsafe, error.size)
    end

    def prepare_native_plans!(handles : Array(UInt64)) : Nil
      unless handles.size == 4 && handles.uniq.size == 4
        raise ArgumentError.new("CogniGraph setup requires four distinct PixelUMM native plans")
      end
      handles.each_with_index do |handle, index|
        error = Bytes.new(1024, 0_u8)
        status = PixelUMMDecoderBlockMPSFFI.prepare_native_islands(
          owner_handle_pointer(handle), ML::Metal::CogniGraphSchedule.callback,
          error.to_unsafe, error.size)
        unless status == 0
          raise ArgumentError.new("PixelUMM CogniGraph preparation failed for plan #{index} (#{status}): #{ffi_error_text(error)}")
        end
      end
    end

    def set_native_plans!(handles : Array(UInt64), enabled : Bool) : Nil
      unless handles.size == 4 && handles.uniq.size == 4
        raise ArgumentError.new("CogniGraph mode change requires four distinct PixelUMM native plans")
      end
      handles.each_with_index do |handle, index|
        error = Bytes.new(1024, 0_u8)
        status = PixelUMMDecoderBlockMPSFFI.set_native_islands(
          owner_handle_pointer(handle), enabled ? 1 : 0, error.to_unsafe, error.size)
        unless status == 0
          raise ArgumentError.new("PixelUMM CogniGraph selection failed for plan #{index} (#{status}): #{ffi_error_text(error)}")
        end
      end
    end

    def native_plan_stats(handles : Array(UInt64)) : Array(NativeStats)
      unless handles.size == 4 && handles.uniq.size == 4
        raise ArgumentError.new("CogniGraph stats require four distinct PixelUMM native plans")
      end
      rows = handles.map_with_index do |handle, index|
        values = Array(UInt64).new(8, 0_u64)
        status = PixelUMMDecoderBlockMPSFFI.native_island_stats(
          owner_handle_pointer(handle), values.to_unsafe, values.size.to_u32)
        unless status == 0
          raise ArgumentError.new("PixelUMM CogniGraph stats failed for plan #{index} (#{status})")
        end
        {graphs: values[0], ops: values[1], waves: values[2], barriers: values[3],
         maxwidth: values[4], native_dispatches_perblock: values[5],
         baseline_encoders_perblock: values[6],
         candidate_encoders_perblock_no_captures: values[7]}
      end
      validate_native_stats!(rows)
      rows
    end
  {% else %}
    def native_prepare_status(handle : UInt64) : Int32
      raise ArgumentError.new("PixelUMM CogniGraph native preparation requires macOS Metal")
    end

    def native_set_status(handle : UInt64, enabled : Int32) : Int32
      raise ArgumentError.new("PixelUMM CogniGraph native selection requires macOS Metal")
    end

    def prepare_native_plans!(handles : Array(UInt64)) : Nil
      raise ArgumentError.new("PixelUMM CogniGraph native preparation requires macOS Metal")
    end

    def set_native_plans!(handles : Array(UInt64), enabled : Bool) : Nil
      raise ArgumentError.new("PixelUMM CogniGraph native selection requires macOS Metal")
    end

    def native_plan_stats(handles : Array(UInt64)) : Array(NativeStats)
      raise ArgumentError.new("PixelUMM CogniGraph native stats require macOS Metal")
    end
  {% end %}

  def validate_native_stats!(rows : Array(NativeStats)) : Nil
    unless rows.size == 4 && rows.uniq.size == 1
      raise ArgumentError.new("CogniGraph plan metadata differs across the four identical block slots")
    end
    row = rows.first
    unless row[:graphs] > 0 && row[:ops] > 0 && row[:waves] > 0 && row[:maxwidth] > 0 &&
           row[:native_dispatches_perblock] > 0 && row[:baseline_encoders_perblock] > 0 &&
           row[:candidate_encoders_perblock_no_captures] > 0 &&
           row[:candidate_encoders_perblock_no_captures] < row[:baseline_encoders_perblock]
      raise ArgumentError.new("CogniGraph native encoder count model is malformed")
    end
  end

  def validate_negative_controls!(controls : NamedTuple(unprepared_enable: Int32,
                                    duplicate_prepare: Int32, invalid_enabled: Int32, late_prepare: Int32)) : Nil
    unless controls.values.all? { |status| status != 0 }
      raise ArgumentError.new("CogniGraph rejected-operation negative control unexpectedly succeeded")
    end
  end

  def metric_json(metric : B::Metric)
    {failures: metric[:failures], different_words: metric[:different_words],
     nonfinite: metric[:nonfinite], elements: metric[:elements]}
  end

  def stats_json(rows : Array(NativeStats))
    rows.map do |row|
      {graphs: row[:graphs], ops: row[:ops], waves: row[:waves],
       barriers: row[:barriers], maxwidth: row[:maxwidth],
       native_dispatches_perblock: row[:native_dispatches_perblock],
       baseline_encoders_perblock: row[:baseline_encoders_perblock],
       candidate_encoders_perblock_no_captures: row[:candidate_encoders_perblock_no_captures]}
    end
  end
end
