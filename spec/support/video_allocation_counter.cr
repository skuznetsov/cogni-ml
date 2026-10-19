# Test-only instrumentation: count every tracked MetalBuffer allocation,
# including allocations released before the end of an encode. Live/peak
# gauges alone cannot establish the absence of temporary allocations.
require "../../src/ml/core/buffer"

class ML::MetalBuffer
  @@video_spec_allocations : Int64 = 0

  def self.video_spec_allocations : Int64
    @@video_spec_allocations
  end

  private def track_alloc(bytes : Int64) : Nil
    @@video_spec_allocations += 1
    previous_def(bytes)
  end
end
