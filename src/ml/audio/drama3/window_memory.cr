require "./native"

module ML::Audio::Drama3
  # OS/driver observations, never a reservation or a physical peak bound.
  # Call at input/terminal boundaries; no GPU synchronization is performed.
  module WindowMemory
    extend self

    {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
      lib Platform
        fun snapshot = gs_drama3_memory_snapshot(physical : UInt64*, free_reclaimable : UInt64*, footprint : UInt64*, peak_footprint : UInt64*) : Int32
      end

      def snapshot
        physical = free_reclaimable = footprint = peak = 0_u64
        valid = Platform.snapshot(pointerof(physical), pointerof(free_reclaimable), pointerof(footprint), pointerof(peak)) == 1
        device = ML::Metal::Device.instance
        {
          valid: valid, physical_bytes: physical.to_i64,
          free_reclaimable_bytes: free_reclaimable.to_i64,
          process_footprint_bytes: footprint.to_i64, process_peak_footprint_bytes: peak.to_i64,
          metal_allocated_bytes: device.current_allocated_size,
          metal_recommended_bytes: device.recommended_working_set_size,
          tracked_buffers: ML::MetalBuffer.stats,
        }
      end
    {% end %}
  end
end
