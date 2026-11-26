require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module Drama3LateCompareSpec
  include ML::Audio::Drama3

  alias Buffer = Resident::Buffer
  alias Context = Resident::Context
  WORDS_PER_RECEIPT = 26
  FLOATS_PER_RECEIPT = WORDS_PER_RECEIPT * 2
  SOURCE = {{ read_file("#{__DIR__}/../examples/support/drama3_late_compare.metal") }}

  def self.enqueue(context : Context, output : Buffer, slot : Int32,
                   actual_values : Array(Float32), reference_values : Array(Float32),
                   atol : Float32, rtol : Float32, offset : UInt64 = 0_u64,
                   route_values : Array(Int32)? = nil, hidden : Int32 = 1) : Nil
    raise "comparator test input length" unless actual_values.size == reference_values.size
    raise "comparator test route length" if route_values && route_values.size != actual_values.size // hidden

    actual = context.upload(actual_values)
    reference = context.upload(reference_values)
    route_ids = if route_values
                  context.upload(route_values.map { |id| id.unsafe_as(Float32) })
                else
                  context.upload([0_f32])
                end
    parts_count = (actual_values.size + 1023) // 1024
    parts = context.alloc(parts_count * WORDS_PER_RECEIPT)
    receipt = output.slice(slot * FLOATS_PER_RECEIPT, FLOATS_PER_RECEIPT)
    params = [
      actual_values.size.to_u64,
      atol.unsafe_as(UInt32).to_u64,
      rtol.unsafe_as(UInt32).to_u64,
      offset,
      route_values ? 1_u64 : 0_u64,
      hidden.to_u64,
    ]
    context.dispatch("dr3_late_compare_tile", SOURCE,
      [actual, reference, parts, route_ids], params, parts_count, 1)
    context.dispatch("dr3_late_compare_reduce", SOURCE,
      [parts, receipt], [parts_count.to_u64], 1, 1)
  end

  def self.word(values : Array(Float32), slot : Int32, word_index : Int32) : UInt32
    base = slot * FLOATS_PER_RECEIPT + word_index * 2
    values[base].to_u32 | (values[base + 1].to_u32 << 16)
  end

  def self.float_word(values : Array(Float32), slot : Int32, word_index : Int32) : Float32
    word(values, slot, word_index).unsafe_as(Float32)
  end

  def self.witness_index(values : Array(Float32), slot : Int32, start : Int32) : UInt64?
    low = word(values, slot, start)
    high = word(values, slot, start + 1)
    return nil if low == UInt32::MAX && high == UInt32::MAX
    (high.to_u64 << 32) | low.to_u64
  end
end

describe "DRAMA-3 late comparator Metal" do
  it "checks exact bits, tolerance boundaries, non-finites, tile edges, and merged witnesses in one terminal read" do
    previous_backend = ENV["DRAMA3_BACKEND"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?

      context = Drama3LateCompareSpec::Context.new
      # Four independent raw receipts, followed by two merged receipts.
      output = context.alloc(6 * Drama3LateCompareSpec::FLOATS_PER_RECEIPT)

      Drama3LateCompareSpec.enqueue(context, output, 0,
        [1_f32, -2_f32, 0_f32, 4_f32], [1_f32, -2_f32, 0_f32, 4_f32],
        6e-5_f32, 3e-4_f32)

      atol = 6e-5_f32
      next_after_atol = (atol.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
      Drama3LateCompareSpec.enqueue(context, output, 1,
        [atol / 2_f32, atol, next_after_atol], [0_f32, 0_f32, 0_f32],
        atol, 3e-4_f32)

      Drama3LateCompareSpec.enqueue(context, output, 2,
        [Float32::NAN, Float32::INFINITY, -Float32::INFINITY, 2_f32],
        [0_f32, Float32::INFINITY, 1_f32, 2_f32], 6e-5_f32, 3e-4_f32)

      boundary_count = 1030
      boundary_actual = Array(Float32).new(boundary_count, 0_f32)
      boundary_reference = Array(Float32).new(boundary_count, 0_f32)
      boundary_routes = Array(Int32).new(boundary_count, 0_i32)
      # Exercise the final item of tile 0, first item of tile 1, and the final
      # item of the short tail in tile 1.
      boundary_actual[1023] = 1e-8_f32
      boundary_actual[1024] = 8e-5_f32
      boundary_actual[1029] = 2e-8_f32
      boundary_routes[1023] = 31
      boundary_routes[1024] = 42
      boundary_routes[1029] = 58
      large_offset = (1_u64 << 32) + 17_u64
      Drama3LateCompareSpec.enqueue(context, output, 3,
        boundary_actual, boundary_reference, 6e-5_f32, 3e-4_f32,
        large_offset, boundary_routes)

      # Slot order is intentionally [3, 0], not ascending. The all-match
      # receipt must not replace the nonzero 64-bit witness with its sentinel.
      context.dispatch("dr3_late_compare_merge", Drama3LateCompareSpec::SOURCE,
        [output, output.slice(4 * Drama3LateCompareSpec::FLOATS_PER_RECEIPT,
          Drama3LateCompareSpec::FLOATS_PER_RECEIPT)],
        [2_u64, 3_u64, 0_u64], 1, 1)
      context.dispatch("dr3_late_compare_merge", Drama3LateCompareSpec::SOURCE,
        [output, output.slice(5 * Drama3LateCompareSpec::FLOATS_PER_RECEIPT,
          Drama3LateCompareSpec::FLOATS_PER_RECEIPT)],
        [1_u64, 0_u64], 1, 1)

      context.readbacks.should be_empty
      values = context.read(output, "terminal")
      context.readbacks.should eq({"terminal" => output.count.to_i64 * 4_i64})

      # All-match receipt: zero counts and absent-witness sentinel.
      Drama3LateCompareSpec.word(values, 0, 0).should eq(0_u32)
      Drama3LateCompareSpec.word(values, 0, 1).should eq(0_u32)
      Drama3LateCompareSpec.word(values, 0, 2).should eq(0_u32)
      Drama3LateCompareSpec.witness_index(values, 0, 5).should be_nil
      Drama3LateCompareSpec.witness_index(values, 0, 12).should be_nil
      Drama3LateCompareSpec.witness_index(values, 0, 19).should be_nil

      # Exact atol passes; the next representable value fails. Bit mismatch
      # begins earlier than the first tolerance failure.
      Drama3LateCompareSpec.word(values, 1, 0).should eq(3_u32)
      Drama3LateCompareSpec.word(values, 1, 1).should eq(1_u32)
      Drama3LateCompareSpec.witness_index(values, 1, 5).should eq(0_u64)
      Drama3LateCompareSpec.witness_index(values, 1, 12).should eq(2_u64)
      Drama3LateCompareSpec.float_word(values, 1, 3).should eq(next_after_atol)
      Drama3LateCompareSpec.float_word(values, 1, 4).should be > 1_f32

      # Non-finite elements have their own count and are excluded from the
      # finite tolerance/max reductions.
      Drama3LateCompareSpec.word(values, 2, 2).should eq(3_u32)
      Drama3LateCompareSpec.word(values, 2, 1).should eq(0_u32)
      Drama3LateCompareSpec.witness_index(values, 2, 19).should eq(0_u64)
      Drama3LateCompareSpec.float_word(values, 2, 3).should eq(0_f32)
      Drama3LateCompareSpec.float_word(values, 2, 4).should eq(0_f32)

      # Counts prove both tile-edge and short-tail elements were included.
      Drama3LateCompareSpec.word(values, 3, 0).should eq(3_u32)
      Drama3LateCompareSpec.word(values, 3, 1).should eq(1_u32)
      Drama3LateCompareSpec.witness_index(values, 3, 5).should eq(large_offset + 1023_u64)
      Drama3LateCompareSpec.word(values, 3, 11).should eq(31_u32)
      Drama3LateCompareSpec.witness_index(values, 3, 12).should eq(large_offset + 1024_u64)
      Drama3LateCompareSpec.word(values, 3, 18).should eq(42_u32)
      Drama3LateCompareSpec.float_word(values, 3, 3).should eq(8e-5_f32)

      # GPU merge preserves nonzero 64-bit witnesses and route IDs despite the
      # out-of-order all-match input; merging only an empty receipt stays null.
      Drama3LateCompareSpec.word(values, 4, 0).should eq(3_u32)
      Drama3LateCompareSpec.word(values, 4, 1).should eq(1_u32)
      Drama3LateCompareSpec.witness_index(values, 4, 5).should eq(large_offset + 1023_u64)
      Drama3LateCompareSpec.word(values, 4, 11).should eq(31_u32)
      Drama3LateCompareSpec.witness_index(values, 4, 12).should eq(large_offset + 1024_u64)
      Drama3LateCompareSpec.word(values, 4, 18).should eq(42_u32)
      Drama3LateCompareSpec.word(values, 5, 0).should eq(0_u32)
      Drama3LateCompareSpec.witness_index(values, 5, 5).should be_nil
      Drama3LateCompareSpec.witness_index(values, 5, 12).should be_nil
      Drama3LateCompareSpec.witness_index(values, 5, 19).should be_nil
    ensure
      if previous_backend
        ENV["DRAMA3_BACKEND"] = previous_backend
      else
        ENV.delete("DRAMA3_BACKEND")
      end
    end
  end
end
{% end %}
