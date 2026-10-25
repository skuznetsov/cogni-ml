require "../../src/ml/core/buffer"

# Diagnostic counters include released temporary named allocations; opaque MPS
# allocations are covered by the external process guard, not these counters.
class ML::MetalBuffer
  @@hy_text_probe_allocations = 0_i64
  @@hy_text_probe_readbacks = 0_i64

  def self.hy_text_probe_allocations : Int64
    @@hy_text_probe_allocations
  end

  def self.hy_text_probe_readbacks : Int64
    @@hy_text_probe_readbacks
  end

  def read_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
    @@hy_text_probe_readbacks += 1
    previous_def(ptr, byte_size)
  end

  def read(count : Int32) : Array(Float32)
    @@hy_text_probe_readbacks += 1
    previous_def(count)
  end

  def read_to_slice(dest : Slice(Float32)) : Nil
    @@hy_text_probe_readbacks += 1
    previous_def(dest)
  end

  private def track_alloc(bytes : Int64) : Nil
    @@hy_text_probe_allocations += 1
    previous_def(bytes)
  end
end
