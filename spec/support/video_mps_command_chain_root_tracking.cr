lib MPSCommandChainRootTrackingProbe
  fun run = gs_mps_chain_root_tracking_probe(unique_roots : Int32,
    retained_roots : Int32*, equality_calls : UInt64*) : Int32
  fun retention = gs_mps_chain_retention_probe(completed_roots : Int32,
    summary_count : Int32*, pending_after_finish : Int32*, max_pending : Int32*,
    error_count : Int32*, callback_records : Int32*, fallback_records : Int32*,
    gpu_seconds : Float64*, released_completed_roots : Int32*,
    tracker_released : Int32*) : Int32
end

module VideoMPSCommandChainRootTrackingProbe
  def self.run(unique_roots : Int32) : Tuple(Int32, Int32, UInt64)
    retained_roots = 0_i32
    equality_calls = 0_u64
    status = MPSCommandChainRootTrackingProbe.run(unique_roots,
      pointerof(retained_roots), pointerof(equality_calls))
    {status, retained_roots, equality_calls}
  end

  def self.retention(completed_roots : Int32) : Tuple(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Float64, Int32, Int32)
    summary_count = 0_i32
    pending_after_finish = 0_i32
    max_pending = 0_i32
    error_count = 0_i32
    callback_records = 0_i32
    fallback_records = 0_i32
    gpu_seconds = 0.0
    released_completed_roots = 0_i32
    tracker_released = 0_i32
    status = MPSCommandChainRootTrackingProbe.retention(completed_roots,
      pointerof(summary_count), pointerof(pending_after_finish),
      pointerof(max_pending), pointerof(error_count),
      pointerof(callback_records), pointerof(fallback_records),
      pointerof(gpu_seconds), pointerof(released_completed_roots),
      pointerof(tracker_released))
    {status, summary_count, pending_after_finish, max_pending, error_count,
      callback_records, fallback_records, gpu_seconds,
      released_completed_roots, tracker_released}
  end
end
