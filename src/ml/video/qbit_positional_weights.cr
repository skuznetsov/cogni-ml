require "../metal/dispatch"
require "./metal_ops"

module ML::Video::QBitPositionalWeights
  # Opt-in, caller-owned weight-codec experiment. Normalization is always 256
  # values; tile_values only controls the addressable packing record. Source
  # Gaussian p4/p5 payloads use the public codec's unmodified means and planes.
  # All operations append to the caller's SERIAL encoder, without allocation,
  # completion waits, or buffer-content reads. Keep every buffer alive through
  # the caller's terminal completion boundary.
  SOURCE       = {{ read_file("#{__DIR__}/kernels/qbit_positional_weights.metal") }}
  STATUS_WORDS = 8
  PLAN_WORDS   = 4

  # status: used blob bytes, p4/p5/BF16 tile counts, correction count,
  # invalid packing tiles, capacity failures, invalid decoding tiles.
  # A nonzero error counter invalidates the operation. decode writes quiet NaN
  # to every output value of an invalid record instead of using partial data.
  @@zero : ML::Metal::ComputePipeline?
  @@plan : ML::Metal::ComputePipeline?
  @@scan : ML::Metal::ComputePipeline?
  @@scan_groups : ML::Metal::ComputePipeline?
  @@scan_finish : ML::Metal::ComputePipeline?
  @@pack : ML::Metal::ComputePipeline?
  @@decode : ML::Metal::ComputePipeline?

  def self.prepare! : Nil
    @@zero ||= pipeline("wan_positional_zero")
    @@plan ||= pipeline("wan_positional_plan")
    @@scan ||= pipeline("wan_positional_scan")
    @@scan_groups ||= pipeline("wan_positional_scan_groups")
    @@scan_finish ||= pipeline("wan_positional_scan_finish")
    @@pack ||= pipeline("wan_positional_pack")
    @@decode ||= pipeline("wan_positional_decode")
  end

  # The blob's worst-case capacity is count*2 + tiles*4, because selection
  # never exceeds the original BF16 record cost. It can also be a smaller
  # planned capacity: GPU bounds checks reject overflow, without a host read.
  # The prefix-offset table occupies another (tiles+1)*4 bytes. Logical used
  # bytes and reserved buffer capacity are different memory measurements.
  def self.pack(e : ML::Metal::ComputeEncoder,
                original_bf16 : ML::MetalBuffer,
                p4 : ML::MetalBuffer, p5 : ML::MetalBuffer,
                lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                plans : ML::MetalBuffer, offsets : ML::MetalBuffer,
                scan : ML::MetalBuffer, blob : ML::MetalBuffer,
                status : ML::MetalBuffer, count : Int32,
                tile_values : Int32, threshold : Float32,
                value_mode : Int32) : Nil
    tiles = geometry!(count, tile_values)
    unless threshold.finite? && threshold >= 0 && (value_mode == 0 || value_mode == 1)
      raise ArgumentError.new("Invalid positional reconstruction policy or value mode")
    end
    bytes!(original_bf16, count.to_i64 * 2)
    bytes!(p4, (count // 256).to_i64 * 136)
    bytes!(p5, (count // 256).to_i64 * 168)
    bytes!(lut4, 1024_i64); bytes!(lut5, 1024_i64)
    bytes!(plans, tiles.to_i64 * PLAN_WORDS * 4)
    bytes!(offsets, (tiles.to_i64 + 1) * 4)
    bytes!(scan, ((tiles.to_i64 + 255) // 256) * 4)
    capacity!(blob)
    bytes!(status, STATUS_WORDS.to_i64 * 4)
    # No mutable workspace may alias another workspace or any resident input.
    inputs = {original_bf16, p4, p5, lut4, lut5}
    work = {plans, offsets, scan, blob, status}
    work.each_with_index do |workspace, i|
      inputs.each { |input| MetalOps.distinct(workspace, input) }
      work.each_with_index { |other, j| MetalOps.distinct(workspace, other) if j > i }
    end
    zero = @@zero || raise ArgumentError.new("Call QBitPositionalWeights.prepare! before encode")
    plan = @@plan.not_nil!; scan_pipeline = @@scan.not_nil!
    scan_groups = @@scan_groups.not_nil!; scan_finish = @@scan_finish.not_nil!
    pack = @@pack.not_nil!
    groups = (tiles + 255) // 256
    e.set_pipeline(zero).set_buffer(status, 0, ML::Metal::BufferAccess::Write).dispatch_1d(STATUS_WORDS)
    e.set_pipeline(plan).set_buffer(original_bf16, 0).set_buffer(p4, 1).set_buffer(p5, 2)
      .set_buffer(lut4, 3).set_buffer(lut5, 4).set_buffer(plans, 5, ML::Metal::BufferAccess::Write)
      .set_value(tile_values.to_u32, 6).set_value(threshold, 7)
      .dispatch_threadgroups({tiles, 1, 1}, {256, 1, 1})
    e.set_pipeline(scan_pipeline).set_buffer(plans, 0)
      .set_buffer(offsets, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(scan, 2, ML::Metal::BufferAccess::Write).set_value(tiles.to_u32, 3)
      .dispatch_threadgroups({groups, 1, 1}, {256, 1, 1})
    e.set_pipeline(scan_groups).set_buffer(scan, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(offsets, 1, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_value(groups.to_u32, 3).set_value(tiles.to_u32, 4).dispatch_1d(1)
    e.set_pipeline(scan_finish).set_buffer(scan, 0)
      .set_buffer(offsets, 1, ML::Metal::BufferAccess::ReadWrite).set_value(tiles.to_u32, 2)
      .dispatch_1d(tiles)
    e.set_pipeline(pack).set_buffer(original_bf16, 0).set_buffer(p4, 1).set_buffer(p5, 2)
      .set_buffer(lut4, 3).set_buffer(lut5, 4).set_buffer(plans, 5).set_buffer(offsets, 6)
      .set_buffer(blob, 7, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 8, ML::Metal::BufferAccess::ReadWrite)
      .set_value(tile_values.to_u32, 9).set_value(threshold, 10)
      .set_value(value_mode.to_u32, 11).set_value(blob.size.to_u32, 12)
      .dispatch_threadgroups({tiles, 1, 1}, {256, 1, 1})
  end

  # Validates offsets, exact lengths, header, finite reconstruction data,
  # strictly increasing positions, and zero padding on GPU before consumption.
  # status is caller initialized (or already initialized by pack).
  def self.decode(e : ML::Metal::ComputeEncoder,
                  offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                  lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                  destination : ML::MetalBuffer, status : ML::MetalBuffer,
                  count : Int32, tile_values : Int32) : Nil
    tiles = geometry!(count, tile_values)
    bytes!(offsets, (tiles.to_i64 + 1) * 4); capacity!(blob)
    bytes!(lut4, 1024_i64); bytes!(lut5, 1024_i64)
    bytes!(destination, count.to_i64 * 4); bytes!(status, STATUS_WORDS.to_i64 * 4)
    {offsets, blob, lut4, lut5}.each do |input|
      MetalOps.distinct(destination, input); MetalOps.distinct(status, input)
    end
    MetalOps.distinct(destination, status)
    selected = @@decode || raise ArgumentError.new("Call QBitPositionalWeights.prepare! before encode")
    e.set_pipeline(selected).set_buffer(offsets, 0).set_buffer(blob, 1)
      .set_buffer(lut4, 2).set_buffer(lut5, 3)
      .set_buffer(destination, 4, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 5, ML::Metal::BufferAccess::ReadWrite)
      .set_value(tile_values.to_u32, 6).set_value(blob.size.to_u32, 7)
      .dispatch_threadgroups({tiles, 1, 1}, {256, 1, 1})
  end

  private def self.geometry!(count : Int32, tile_values : Int32) : Int32
    unless {256, 512, 1024}.includes?(tile_values) && count > 0 && count % tile_values == 0
      raise ArgumentError.new("Positional count must contain whole 256/512/1024-value tiles")
    end
    # Includes the largest possible BF16 blob and prevents UInt32 scan overflow.
    tiles = count // tile_values
    if count.to_i64 * 2 + tiles.to_i64 * 4 > UInt32::MAX
      raise ArgumentError.new("Positional payload exceeds UInt32 offsets")
    end
    tiles
  end

  private def self.bytes!(buffer : ML::MetalBuffer, minimum : Int64) : Nil
    unless buffer.valid? && minimum > 0 && buffer.size >= minimum
      raise ArgumentError.new("Invalid or undersized positional weight buffer")
    end
  end

  private def self.capacity!(buffer : ML::MetalBuffer) : Nil
    bytes!(buffer, 4_i64)
    if buffer.size > UInt32::MAX
      raise ArgumentError.new("Positional blob capacity exceeds UInt32")
    end
  end

  private def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end
end
