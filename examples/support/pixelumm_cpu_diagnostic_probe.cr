require "file/tempfile"
require "./pixelumm_decoder_probe"

# Host-side snapshot handling for the PixelUMM36 + head CPU diagnostic.
# The schedule changes only when saved BF16 captures are compared; it does not
# select a GPU graph, input, baseline, or profiler mode.
module PixelUMMCPUDiagnosticProbe
  extend self

  CAPTURE_REQUESTS = {0 => 1, 1 => 3}
  REQUEST_COUNT    = 6

  alias Comparison = PixelUMMHeadProbe::Comparison

  record Capture, stage : String, nbytes : Int64
  record Arm, kind : String, role : String, policy : String,
    snapshot_slot : Int32? = nil, preceding_policy : String? = nil

  def schedule(reverse : Bool = false) : Array(Arm)
    condition_policies = reverse ? {"deferred", "eager"} : {"eager", "deferred"}
    [
      Arm.new("warm", "warm", "standard"),
      Arm.new("condition_0", "condition", condition_policies[0], 0),
      Arm.new("probe_0", "probe", "standard", nil, condition_policies[0]),
      Arm.new("condition_1", "condition", condition_policies[1], 1),
      Arm.new("probe_1", "probe", "standard", nil, condition_policies[1]),
      Arm.new("tail", "tail", "standard"),
    ]
  end

  # The file-backed snapshots retain actual captures after the live GPU buffers
  # are reused. Comparisons use the same preallocated CPU scratch in both arms.
  class Store
    getter total_bytes : Int64
    getter per_slot_bytes : Int64

    enum SlotState
      Pending
      Capturing
      Complete
      Failed
    end

    @paths : Array(String)
    @captures : Array(Capture)
    @scratch : PixelUMMTwoBlockProbe::ComparisonScratch
    @files : Array(File)
    @states : Array(SlotState)
    @requests : Array(Int32?)
    @compare_now : Array(Bool?)
    @comparison_started : Array(Bool)
    @comparison_done : Array(Bool)
    @requests_finished : Bool
    @released : Bool

    def initialize(paths : Array(String), captures : Array(Capture),
                   @scratch : PixelUMMTwoBlockProbe::ComparisonScratch)
      @total_bytes = 0_i64
      @per_slot_bytes = 0_i64
      @paths = [] of String
      @captures = [] of Capture
      @files = [] of File
      @states = Array(SlotState).new(2, SlotState::Pending)
      @requests = Array(Int32?).new(2, nil)
      @compare_now = Array(Bool?).new(2, nil)
      @comparison_started = Array(Bool).new(2, false)
      @comparison_done = Array(Bool).new(2, false)
      @requests_finished = false
      @released = false

      validate_inventory!(paths, captures)
      @captures = captures.dup
      create_snapshot_files!
    end

    def paths : Array(String)
      @paths.dup
    end

    def captures : Array(Capture)
      @captures.dup
    end

    def slot_completed?(slot : Int32) : Bool
      validate_slot!(slot)
      @states[slot] == SlotState::Complete
    end

    def slot_request(slot : Int32) : Int32?
      validate_slot!(slot)
      @requests[slot]
    end

    def slot_policy(slot : Int32) : String?
      validate_slot!(slot)
      value = @compare_now[slot]
      value.nil? ? nil : (value ? "eager" : "deferred")
    end

    def completed? : Bool
      @requests_finished &&
        @states.all? { |state| state == SlotState::Complete } &&
        @comparison_done.all?
    end

    # Call once all six GPU requests have terminated. This explicit barrier
    # keeps deferred comparisons outside the inference/request loop.
    def finish_requests!(completed_requests : Int32) : Nil
      ensure_open!
      raise ArgumentError.new("CPU diagnostic requires all six GPU requests before deferred comparison") unless completed_requests == REQUEST_COUNT
      unless @states.all? { |state| state == SlotState::Complete } &&
             @compare_now.compact.uniq.size == 2
        raise ArgumentError.new("CPU diagnostic requires one completed eager and one completed deferred snapshot")
      end
      raise ArgumentError.new("CPU diagnostic request completion was already recorded") if @requests_finished
      @requests_finished = true
    end

    def snapshot(slot : Int32, request : Int32, compare_now : Bool,
                 expected_reader : Proc(String, Bytes, Nil),
                 actual_reader : Proc(String, Bytes, Nil)) : Array(Comparison)
      ensure_open!
      validate_slot!(slot)
      expected_request = CAPTURE_REQUESTS[slot]
      unless request == expected_request
        raise ArgumentError.new("CPU diagnostic snapshot slot/request mismatch")
      end
      raise ArgumentError.new("CPU diagnostic snapshot store has a failed slot") if @states.any? { |state| state == SlotState::Failed }
      raise ArgumentError.new("CPU diagnostic snapshots must follow request order") unless request == next_snapshot_request
      raise ArgumentError.new("CPU diagnostic snapshots are closed after request completion") if @requests_finished
      raise ArgumentError.new("CPU diagnostic snapshot slot cannot be overwritten") unless @states[slot] == SlotState::Pending
      ensure_file_extent!(slot)

      @states[slot] = SlotState::Capturing
      @requests[slot] = request
      @compare_now[slot] = compare_now
      begin
        offset = 0_i64
        file = @files[slot]
        @captures.each do |capture|
          _, actual = @scratch.buffers(capture.nbytes)
          actual_reader.call(capture.stage, actual)
          file.seek(offset, IO::Seek::Set)
          file.write(actual)
          offset = checked_add(offset, capture.nbytes, "snapshot offset")
        end
        file.flush
        ensure_file_extent!(slot)
        @states[slot] = SlotState::Complete
        return [] of Comparison unless compare_now

        compare_saved(slot, expected_reader)
      rescue ex
        @states[slot] = SlotState::Failed
        raise ex
      end
    end

    def compare_deferred(slot : Int32,
                         expected_reader : Proc(String, Bytes, Nil)) : Array(Comparison)
      ensure_open!
      validate_slot!(slot)
      raise ArgumentError.new("CPU diagnostic deferred comparison is premature") unless @requests_finished
      raise ArgumentError.new("CPU diagnostic slot has no completed snapshot") unless @states[slot] == SlotState::Complete
      raise ArgumentError.new("CPU diagnostic slot is not deferred") unless @compare_now[slot] == false
      raise ArgumentError.new("CPU diagnostic deferred slot was already compared") if @comparison_started[slot]

      compare_saved(slot, expected_reader)
    end

    # Closing releases descriptors only. Snapshot files are retained as logs.
    def close : Nil
      return if @released
      @files.each do |file|
        file.close unless file.closed?
      end
      @released = true
    end

    def release : Nil
      close
    end

    private def compare_saved(slot : Int32,
                              expected_reader : Proc(String, Bytes, Nil)) : Array(Comparison)
      raise ArgumentError.new("CPU diagnostic slot was already compared") if @comparison_started[slot]
      @comparison_started[slot] = true
      ensure_file_extent!(slot)
      file = @files[slot]
      comparisons = [] of Comparison
      offset = 0_i64
      begin
        @captures.each do |capture|
          expected, actual = @scratch.buffers(capture.nbytes)
          file.seek(offset, IO::Seek::Set)
          begin
            file.read_fully(actual)
          rescue ex : IO::EOFError
            raise ArgumentError.new("CPU diagnostic snapshot file is truncated")
          end
          expected_reader.call(capture.stage, expected)
          comparisons << PixelUMMHeadProbe.compare_bytes(capture.stage, expected, actual)
          offset = checked_add(offset, capture.nbytes, "comparison offset")
        end
        ensure_file_extent!(slot)
        @comparison_done[slot] = true
        comparisons
      rescue ex
        raise ex
      end
    end

    private def next_snapshot_request : Int32
      return 1 if @states[0] == SlotState::Pending
      return 3 if @states[1] == SlotState::Pending
      REQUEST_COUNT
    end

    private def ensure_open! : Nil
      raise ArgumentError.new("CPU diagnostic snapshot store is closed") if @released
    end

    private def validate_slot!(slot : Int32) : Nil
      raise ArgumentError.new("CPU diagnostic snapshot slot must be zero or one") unless 0 <= slot < 2
    end

    private def ensure_file_extent!(slot : Int32) : Nil
      unless @files[slot].size == @per_slot_bytes
        raise ArgumentError.new("CPU diagnostic snapshot file extent changed")
      end
    end

    private def validate_inventory!(paths : Array(String), captures : Array(Capture)) : Nil
      raise ArgumentError.new("CPU diagnostic store requires exactly two snapshot paths") unless paths.size == 2
      raise ArgumentError.new("CPU diagnostic store requires a nonempty capture inventory") if captures.empty?

      canonical_paths = paths.map do |path|
        raise ArgumentError.new("CPU diagnostic snapshot paths must be absolute") unless Path[path].absolute?
        expanded = File.expand_path(path)
        basename = File.basename(expanded)
        parent = File.dirname(expanded)
        raise ArgumentError.new("CPU diagnostic snapshot parent directory must exist") unless Dir.exists?(parent)
        File.join(File.realpath(parent), basename)
      end
      raise ArgumentError.new("CPU diagnostic snapshot paths must be distinct") unless canonical_paths.uniq.size == 2
      @paths = canonical_paths

      seen = Set(String).new
      total = 0_i64
      max_capture = 0_i64
      captures.each do |capture|
        unless !capture.stage.empty? && seen.add?(capture.stage)
          raise ArgumentError.new("CPU diagnostic capture stages must be nonempty and unique")
        end
        unless 0 < capture.nbytes <= Int32::MAX.to_i64 && capture.nbytes.even?
          raise ArgumentError.new("CPU diagnostic captures must be positive even BF16 extents within Int32")
        end
        total = checked_add(total, capture.nbytes, "capture inventory")
        max_capture = Math.max(max_capture, capture.nbytes)
      end
      raise ArgumentError.new("CPU diagnostic snapshot total overflows") if total > Int64::MAX // 2
      begin
        @scratch.buffers(max_capture)
      rescue ex : ArgumentError
        raise ArgumentError.new("CPU diagnostic comparison scratch capacity is insufficient")
      end
      @per_slot_bytes = total
      @total_bytes = total * 2
    end

    private def create_snapshot_files! : Nil
      created_paths = [] of String
      temp_files = [] of File
      begin
        @paths.each do |path|
          file = File.tempfile("pixelumm-cpu-diagnostic-", ".bf16", dir: File.dirname(path))
          temp_files << file
          file.truncate(@per_slot_bytes)
          file.flush
          File.link(file.path, path)
          created_paths << path
          File.delete(file.path)
          @files << file
        end
      rescue ex
        temp_files.each do |file|
          File.delete?(file.path)
          file.close unless file.closed?
        end
        created_paths.each { |path| File.delete?(path) }
        @files.clear
        raise ArgumentError.new("CPU diagnostic snapshot files must be fresh exclusive files: #{ex.message}")
      end
    end

    private def checked_add(left : Int64, right : Int64, label : String) : Int64
      raise ArgumentError.new("CPU diagnostic #{label} overflows Int64") if left > Int64::MAX - right
      left + right
    end
  end
end
