lib MPSCommandChainRootTrackingProbe
  fun run = gs_mps_chain_root_tracking_probe(unique_roots : Int32,
    retained_roots : Int32*, equality_calls : UInt64*) : Int32
end

module VideoMPSCommandChainRootTrackingProbe
  def self.run(unique_roots : Int32) : Tuple(Int32, Int32, UInt64)
    retained_roots = 0_i32
    equality_calls = 0_u64
    status = MPSCommandChainRootTrackingProbe.run(unique_roots,
      pointerof(retained_roots), pointerof(equality_calls))
    {status, retained_roots, equality_calls}
  end
end
