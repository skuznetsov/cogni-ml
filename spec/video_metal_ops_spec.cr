require "spec"
require "../src/ml/video/metal_ops"
require "./support/video_allocation_counter"

lib LibVideoSpecMemory
  fun posix_memalign(ptr : Void**, alignment : LibC::SizeT, size : LibC::SizeT) : LibC::Int
  fun getpagesize : LibC::Int
  fun free(ptr : Void*) : Void
end

private def video_metal_bf16_expand(value : Float32) : Float32
  (value.unsafe_as(UInt32) & 0xffff0000_u32).unsafe_as(Float32)
end

private def video_metal_resident_weight(
  arena : ML::Video::Arena, name : String, values : Array(Float32),
  dtype : ML::Video::WeightDType,
) : ML::MetalBuffer
  case dtype
  when ML::Video::WeightDType::F32
    arena.upload(name, values)
  when ML::Video::WeightDType::BF16
    buffer = arena.alloc_weight(name, values.size, dtype)
    raw = Bytes.new(values.size * sizeof(UInt16))
    values.each_with_index do |value, index|
      bits = (value.unsafe_as(UInt32) >> 16).to_u16
      raw[index * 2] = (bits & 0xff).to_u8
      raw[index * 2 + 1] = (bits >> 8).to_u8
    end
    buffer.write_bytes(raw.to_unsafe, raw.size)
    buffer
  else
    raise ArgumentError.new("Unsupported resident weight dtype")
  end
end

describe ML::Video::MetalOps do
  it "keeps every projection term around the reduction width and its tails" do
    [15, 16, 17, 31, 32, 33].each do |dim|
      arena = ML::Video::Arena.new
      begin
        rows, channels = 2, 3
        values = Array(Float32).new(rows * dim) do |i|
          i < dim ? 1.0_f32 : (((i * 7) % 19 - 9) * 0.125).to_f32
        end
        weights = Array(Float32).new(channels * dim) do |i|
          col, lane = i // dim, i % dim
          sign = lane.even? ? 1.0_f32 : -1.0_f32
          sign * (8.0_f32 + col * 0.25_f32 + (lane % 5) * 0.015625_f32)
        end
        biases = [0.03125_f32, -0.0625_f32, 0.125_f32]
        reference = Array(Float64).new(rows * channels) do |i|
          row, col = i // channels, i % channels
          (0...dim).sum(0.0_f64) do |lane|
            values[row * dim + lane].to_f64 * weights[col * dim + lane].to_f64
          end + biases[col].to_f64
        end
        x = arena.upload("x", values)
        w = arena.upload("w", weights)
        b = arena.upload("b", biases)
        output = arena.alloc("output", rows * channels)
        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        2.times { ML::Video::MetalOps.linear(encoder, x, w, b, output, rows, dim, channels) }
        encoder.end_encoding
        command.commit
        command.wait
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        output.read(rows * channels).zip(reference).each do |actual, expected|
          fail "Non-finite linear result at width #{dim}" unless actual.finite?
          actual.should be_close(expected, 2e-5 + 2e-5 * expected.abs)
        end
      ensure
        arena.release
      end
    end
  end

  it "saturates tanh GELU for large finite checkpoint activations" do
    arena = ML::Video::Arena.new
    begin
      values = [-10.0_f32, 10.0_f32, -12.0_f32, 12.0_f32, -100.0_f32,
                100.0_f32, -10000.0_f32, 10000.0_f32, -1e20_f32, 1e20_f32, 0.0_f32]
      input = arena.upload("input", values)
      output = arena.alloc("output", values.size)
      allocations = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      2.times { ML::Video::MetalOps.gelu(encoder, input, output, values.size) }
      encoder.end_encoding
      command.commit
      command.wait
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      output.read(values.size).zip(values).each do |actual, value|
        expected = value > 0 ? value : 0.0_f32
        fail "Non-finite GELU #{actual} at finite input #{value}" unless actual.finite?
        actual.should be_close(expected, 2e-5 + 2e-5 * expected.abs)
      end
    ensure
      arena.release
    end
  end

  it "encodes a projection and GELU without host intermediates or buffer allocations" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", [1.0_f32, -2.0_f32, 0.5_f32, 3.0_f32])
      w = arena.upload("w", [0.5_f32, 2.0_f32, -1.0_f32, 0.25_f32])
      b = arena.upload("b", [0.25_f32, -0.5_f32])
      projected = arena.alloc("projected", 4)
      output = arena.alloc("output", 4)
      count = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      2.times do
        ML::Video::MetalOps.linear(encoder, x, w, b, projected, 2, 2, 2)
        ML::Video::MetalOps.gelu(encoder, projected, output, 4)
      end
      encoder.end_encoding
      command.commit
      command.wait
      ML::MetalBuffer.video_spec_allocations.should eq(count)
      projected.read(4).should eq([-3.25_f32, -2.0_f32, 6.5_f32, -0.25_f32])
      output.read(4).zip(projected.read(4)).each do |actual, value|
        reference = 0.5 * value * (1.0 + Math.tanh(Math.sqrt(2.0 / Math::PI) * (value + 0.044715 * value**3)))
        actual.should be_close(reference, 2e-5 + 2e-5 * reference.abs)
      end
    ensure
      arena.release
    end
  end

  it "rejects unsafe projection aliases and undersized buffers before dispatch" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", [1.0_f32, 2.0_f32])
      w = arena.upload("w", [1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32])
      b = arena.upload("b", [0.0_f32, 0.0_f32])
      small = arena.alloc("small", 1)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) { ML::Video::MetalOps.linear(encoder, x, w, b, x, 1, 2, 2) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.linear(encoder, x, w, b, small, 1, 2, 2) }
      encoder.end_encoding
      command.discard
    ensure
      arena.release
    end
  end

  it "keeps attention batches separate and normalizes each row on GPU" do
    arena = ML::Video::Arena.new
    begin
      q = arena.upload("q", Array(Float32).new(4, 0.0_f32))
      k = arena.upload("k", [0.2_f32, -0.3_f32, 0.7_f32, 0.8_f32,
                             -0.4_f32, 0.1_f32, 0.3_f32, -0.9_f32])
      v = arena.upload("v", [1.0_f32, 2.0_f32, 3.0_f32, 6.0_f32,
                             10.0_f32, 12.0_f32, 14.0_f32, 16.0_f32])
      result_buffer = arena.alloc("result_buffer", 4)
      normalized = arena.alloc("normalized", 4)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      ML::Video::MetalOps.attention(encoder, q, k, v, result_buffer, 1, 2, 1, 2, 2)
      ML::Video::MetalOps.norm(encoder, result_buffer, q, q, normalized, 2, 2, 1e-6_f32)
      encoder.end_encoding
      command.commit
      command.wait
      result_buffer.read(4).should eq([2.0_f32, 4.0_f32, 12.0_f32, 14.0_f32])
      normalized.read(4).zip([-1.0_f32, 1.0_f32, -1.0_f32, 1.0_f32]).each do |a, b|
        a.should be_close(b, 2e-5_f32)
      end
    ensure
      arena.release
    end
  end

  it "rejects invalid broadcast shapes and released buffers" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
      gate = arena.upload("gate", [0.5_f32, 0.25_f32])
      result_buffer = arena.alloc("result_buffer", 4)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.add(encoder, x, x, result_buffer, 4, gate, 3)
      end
      result_buffer.release
      expect_raises(ArgumentError) { ML::Video::MetalOps.copy(encoder, x, result_buffer, 4) }
      encoder.end_encoding
      command.discard
    ensure
      arena.release
    end
  end

  it "matches weighted multi-SIMD attention, RMSNorm, and LayerNorm" do
    arena = ML::Video::Arena.new
    begin
      dim, nq, nk, heads, batches = 80, 2, 3, 2, 2
      qa = Array(Float32).new(batches * nq * heads * dim) { |i| (Math.sin(i * 0.13) * 0.7).to_f32 }
      ka = Array(Float32).new(batches * nk * heads * dim) { |i| (Math.cos(i * 0.09) * 0.6).to_f32 }
      va = Array(Float32).new(ka.size) { |i| (Math.sin(i * 0.07) + i // (nk * heads * dim)).to_f32 }
      weights = Array(Float32).new(dim) { |i| (0.8 + 0.003 * i).to_f32 }
      biases = Array(Float32).new(dim) { |i| (0.01 * Math.sin(i)).to_f32 }
      q, k, v = arena.upload("q", qa), arena.upload("k", ka), arena.upload("v", va)
      w, b = arena.upload("w", weights), arena.upload("b", biases)
      attended = arena.alloc("attended", qa.size)
      normalized = arena.alloc("normalized", qa.size)
      layer_normalized = arena.alloc("layer_normalized", qa.size)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      ML::Video::MetalOps.attention(encoder, q, k, v, attended, nq, nk, heads, dim, batches)
      ML::Video::MetalOps.norm(encoder, attended, w, b, normalized, batches * nq * heads, dim, 1e-6_f32, rms: true, affine: true)
      ML::Video::MetalOps.norm(encoder, attended, w, b, layer_normalized, batches * nq * heads, dim, 1e-6_f32, affine: true)
      encoder.end_encoding
      command.commit
      command.wait
      expected = Array(Float64).new(qa.size, 0.0)
      batches.times do |batch|
        nq.times do |query|
          heads.times do |head|
            qb = ((batch * nq + query) * heads + head) * dim
            scores = Array(Float64).new(nk) do |key|
              kb = ((batch * nk + key) * heads + head) * dim
              dim.times.sum(0.0) { |d| qa[qb + d].to_f64 * ka[kb + d] } / Math.sqrt(dim.to_f64)
            end
            exps = scores.map { |score| Math.exp(score - scores.max) }
            dim.times do |d|
              expected[qb + d] = nk.times.sum(0.0) do |key|
                kb = ((batch * nk + key) * heads + head) * dim
                exps[key] * va[kb + d]
              end / exps.sum
            end
          end
        end
      end
      attended.read(qa.size).zip(expected).each { |a, r| a.should be_close(r, 2e-5 + 2e-5 * r.abs) }
      normalized.read(qa.size).each_slice(dim).with_index do |row, index|
        reference_row = expected[index * dim, dim]
        inv = 1.0 / Math.sqrt(reference_row.sum { |z| z * z } / dim + 1e-6)
        row.each_with_index do |actual, d|
          reference = reference_row[d] * inv * weights[d] + biases[d]
          actual.should be_close(reference, 2e-5 + 2e-5 * reference.abs)
        end
      end
      layer_normalized.read(qa.size).each_slice(dim).with_index do |row, index|
        reference_row = expected[index * dim, dim]
        mean = reference_row.sum / dim
        variance = reference_row.sum { |z| (z - mean)**2 } / dim
        inv = 1.0 / Math.sqrt(variance + 1e-6)
        row.each_with_index do |actual, d|
          reference = (reference_row[d] - mean) * inv * weights[d] + biases[d]
          actual.should be_close(reference, 2e-5 + 2e-5 * reference.abs)
        end
      end
    ensure
      arena.release
    end
  end

  it "rejects overlapping no-copy wrappers even when their Metal handles differ" do
    ML::Metal::Device.init!.should be_true
    page = LibVideoSpecMemory.getpagesize.to_u64
    ptr = Pointer(Void).null
    LibVideoSpecMemory.posix_memalign(pointerof(ptr), page, 3_u64 * page).should eq(0)
    first = second = nil.as(ML::MetalBuffer?)
    begin
      first = ML::MetalBuffer.wrap_no_copy(ptr, (2 * page).to_i64)
      second = ML::MetalBuffer.wrap_no_copy((ptr.as(UInt8*) + page).as(Void*), (2 * page).to_i64)
      first.handle.should_not eq(second.handle)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError, "Partially overlapping video buffers") do
        ML::Video::MetalOps.copy(encoder, first, second, (page * 3 // 8).to_i32)
      end
      expect_raises(ArgumentError, "Unsafe video buffer alias") do
        ML::Video::MetalOps.distinct(second, first)
      end
      encoder.end_encoding
      command.discard
    ensure
      first.try(&.release)
      second.try(&.release)
      LibVideoSpecMemory.free(ptr)
    end
  end

  it "detects temporary allocations even when live-buffer counts return to baseline" do
    ML::Metal::Device.init!.should be_true
    live = ML::MetalBuffer.stats[:live_buffers]
    allocated = ML::MetalBuffer.video_spec_allocations
    temporary = ML::MetalBuffer.new(4_i64)
    temporary.release
    ML::MetalBuffer.stats[:live_buffers].should eq(live)
    ML::MetalBuffer.video_spec_allocations.should eq(allocated + 1)
  end

  it "uses valid dummy bindings for non-affine norm while rejecting released affine weights" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", [-2.0_f32, 0.0_f32, 2.0_f32, 4.0_f32])
      released = arena.alloc("unused", 1)
      released.release
      y = arena.alloc("y", 4)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.norm(encoder, x, released, released, y, 1, 4, 1e-6_f32, affine: true)
      end
      ML::Video::MetalOps.norm(encoder, x, released, released, y, 1, 4, 1e-6_f32)
      encoder.end_encoding
      command.commit
      command.wait
      y.read(4).zip([-3.0, -1.0, 1.0, 3.0]).each do |actual, z|
        reference = z / Math.sqrt(5.0 + 1e-6)
        actual.should be_close(reference, 2e-5 + 2e-5 * reference.abs)
      end
    ensure
      arena.release
    end
  end

  it "tracks resident weight dtype and validates its byte extent" do
    arena = ML::Video::Arena.new
    begin
      ordinary = arena.alloc("ordinary", 2)
      packed = arena.alloc_weight("packed", 3, ML::Video::WeightDType::BF16)
      ordinary.size.should eq(2 * sizeof(Float32))
      packed.size.should eq(3 * sizeof(UInt16))
      arena.dtype("ordinary").should eq(ML::Video::WeightDType::F32)
      arena.dtype("packed").should eq(ML::Video::WeightDType::BF16)
      arena.check_weight("ordinary", 2_i64)
      arena.check_weight("packed", 3_i64)
      ML::Video::MetalOps.check_storage(ordinary, 3_i64, ML::Video::WeightDType::BF16)
      expect_raises(ArgumentError) { arena.check_weight("ordinary", 3_i64) }
      expect_raises(ArgumentError) { arena.check_weight("packed", 4_i64) }
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.check_storage(ordinary, 3_i64, ML::Video::WeightDType::F32)
      end
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.check_storage(packed, 0_i64, ML::Video::WeightDType::BF16)
      end
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.check_storage(
          ordinary, Int32::MAX.to_i64 + 1, ML::Video::WeightDType::BF16)
      end

      released = arena.alloc_weight("released_weight", 1, ML::Video::WeightDType::BF16)
      released.release
      expect_raises(ArgumentError) { arena.check_weight("released_weight", 1_i64) }
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.check_storage(released, 1_i64, ML::Video::WeightDType::BF16)
      end

      removed = arena.alloc_weight("removed_weight", 2, ML::Video::WeightDType::BF16)
      arena.remove("removed_weight")
      removed.valid?.should be_false
      arena.buffers.has_key?("removed_weight").should be_false
      arena.alloc("removed_weight", 1).size.should eq(sizeof(Float32))
      arena.dtype("removed_weight").should eq(ML::Video::WeightDType::F32)
    ensure
      arena.release
    end
  end

  it "rejects replacement of a typed resident weight before the encode preflight" do
    arena = ML::Video::Arena.new
    begin
      original = arena.alloc_weight("typed_weight", 2, ML::Video::WeightDType::BF16)
      replacement = arena.alloc("larger_f32_replacement", 4)
      replacement.size.should be > original.size
      arena.buffers["typed_weight"] = replacement

      expect_raises(ArgumentError, "Resident video weight buffer identity changed") do
        arena.dtype("typed_weight")
      end
      expect_raises(ArgumentError, "Resident video weight buffer identity changed") do
        arena.check_weight("typed_weight", 2_i64)
      end

      # Ordinary F32 scratch remains replaceable for the no-copy alias tests.
      arena.dtype("larger_f32_replacement").should eq(ML::Video::WeightDType::F32)
      arena.buffers["typed_weight"] = original
    ensure
      arena.release
    end
  end

  it "accounts for and releases replaced typed buffers exactly once" do
    arena = ML::Video::Arena.new
    live_bytes = ML::MetalBuffer.stats[:live_bytes]
    begin
      original = arena.alloc_weight("typed_weight", 2, ML::Video::WeightDType::BF16)
      replacement = arena.alloc("f32_replacement", 4)
      arena.buffers["typed_weight"] = replacement

      # The replacement appears under two public keys, while the replaced
      # BF16 allocation remains Arena-owned through typed metadata.
      arena.resident_bytes.should eq(original.size + replacement.size)
      arena.remove("typed_weight")
      original.valid?.should be_false
      replacement.valid?.should be_false
      arena.resident_bytes.should eq(0_i64)
      ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)
    ensure
      arena.release
    end
  end

  it "keeps deleted typed allocations owned until remove or release" do
    arena = ML::Video::Arena.new
    live_bytes = ML::MetalBuffer.stats[:live_bytes]
    original = arena.alloc_weight("deleted_typed_weight", 3, ML::Video::WeightDType::BF16)
    begin
      arena.buffers.delete("deleted_typed_weight").should be(original)
      arena.resident_bytes.should eq(original.size)
      expect_raises(ArgumentError, /Duplicate/) do
        arena.alloc("deleted_typed_weight", 1)
      end
      expect_raises(ArgumentError, /Duplicate/) do
        arena.alloc_weight("deleted_typed_weight", 1, ML::Video::WeightDType::F32)
      end

      arena.remove("deleted_typed_weight")
      original.valid?.should be_false
      arena.resident_bytes.should eq(0_i64)

      # Explicit removal clears the typed owner and permits safe name reuse.
      reused = arena.alloc("deleted_typed_weight", 1)
      arena.dtype("deleted_typed_weight").should eq(ML::Video::WeightDType::F32)
      reused.size.should eq(sizeof(Float32))
    ensure
      arena.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)

    release_arena = ML::Video::Arena.new
    release_live_bytes = ML::MetalBuffer.stats[:live_bytes]
    released_orphan = release_arena.alloc_weight("release_orphan", 2, ML::Video::WeightDType::BF16)
    begin
      release_arena.buffers.delete("release_orphan")
      release_arena.resident_bytes.should eq(released_orphan.size)
    ensure
      release_arena.release
    end
    released_orphan.valid?.should be_false
    release_arena.resident_bytes.should eq(0_i64)
    ML::MetalBuffer.stats[:live_bytes].should eq(release_live_bytes)
  end

  it "keeps BF16 linear weights resident with exact F32 reduction parity across tails and bias types" do
    [15, 16, 17, 31, 32, 33].each do |dim|
      arena = ML::Video::Arena.new
      begin
        rows, channels = 2, 3
        inputs = Array(Float32).new(rows * dim) do |i|
          i < dim ? 1.0_f32 : (((i * 7) % 19 - 9) * 0.125).to_f32
        end
        weights = Array(Float32).new(channels * dim) do |i|
          col, lane = i // dim, i % dim
          sign = lane.even? ? 1.0_f32 : -1.0_f32
          sign * (8.0_f32 + col * 0.25_f32 + (lane % 5) * 0.015625_f32)
        end
        biases = [0.1_f32, -0.3_f32, 0.7_f32]
        x = arena.upload("linear_x", inputs)
        dtype_pairs = {
          {ML::Video::WeightDType::F32, ML::Video::WeightDType::F32},
          {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16},
          {ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32},
          {ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16},
        }
        cases = [] of Tuple(ML::MetalBuffer, ML::MetalBuffer, ML::MetalBuffer,
                            ML::MetalBuffer, Array(Float32), Array(Float32))
        dtype_pairs.each_with_index do |(weight_dtype, bias_dtype), index|
          prefix = "linear_#{index}"
          effective_weights = weight_dtype == ML::Video::WeightDType::BF16 ? weights.map { |value| video_metal_bf16_expand(value) } : weights
          effective_biases = bias_dtype == ML::Video::WeightDType::BF16 ? biases.map { |value| video_metal_bf16_expand(value) } : biases
          w = video_metal_resident_weight(arena, "#{prefix}_weight", weights, weight_dtype)
          b = video_metal_resident_weight(arena, "#{prefix}_bias", biases, bias_dtype)
          ref_w = arena.upload("#{prefix}_reference_weight", effective_weights)
          ref_b = arena.upload("#{prefix}_reference_bias", effective_biases)
          output = arena.alloc("#{prefix}_output", rows * channels)
          reference = arena.alloc("#{prefix}_reference", rows * channels)
          cases << {w, b, output, reference, effective_weights, effective_biases}
        end

        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        dtype_pairs.each_with_index do |(weight_dtype, bias_dtype), index|
          w, b, output, reference, _, _ = cases[index]
          ref_w = arena["linear_#{index}_reference_weight"]
          ref_b = arena["linear_#{index}_reference_bias"]
          ML::Video::MetalOps.linear(encoder, x, w, b, output, rows, dim, channels,
            weight_dtype: weight_dtype, bias_dtype: bias_dtype)
          ML::Video::MetalOps.linear(encoder, x, ref_w, ref_b, reference, rows, dim, channels)
        end
        encoder.end_encoding
        command.commit
        command.wait
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)

        cases.each do |(_, _, output, reference, effective_weights, effective_biases)|
          actual = output.read(rows * channels)
          expected = reference.read(rows * channels)
          actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
          actual.each_with_index do |value, index|
            row, col = index // channels, index % channels
            total = (0...dim).sum(0.0_f64) do |lane|
              inputs[row * dim + lane].to_f64 * effective_weights[col * dim + lane].to_f64
            end + effective_biases[col].to_f64
            value.should be_close(total, 2e-5 + 2e-5 * total.abs)
          end
        end
      ensure
        arena.release
      end
    end
  end

  it "keeps BF16 affine RMSNorm and LayerNorm bit-identical to F32-expanded parameters" do
    arena = ML::Video::Arena.new
    begin
      rows, dim = 2, 37
      inputs = Array(Float32).new(rows * dim) { |i| (Math.sin(i * 0.13) * 2.0).to_f32 }
      weights = Array(Float32).new(dim) { |i| (0.7 + 0.007 * i).to_f32 }
      biases = Array(Float32).new(dim) { |i| (0.13 * Math.sin(i * 0.31)).to_f32 }
      x = arena.upload("norm_x", inputs)
      dtype_pairs = {
        {ML::Video::WeightDType::F32, ML::Video::WeightDType::F32},
        {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16},
        {ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32},
        {ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16},
      }
      cases = [] of Tuple(ML::MetalBuffer, ML::MetalBuffer, Array(ML::MetalBuffer),
                          Array(ML::MetalBuffer), Array(Float32), Array(Float32))
      dtype_pairs.each_with_index do |(weight_dtype, bias_dtype), index|
        prefix = "norm_#{index}"
        effective_weights = weight_dtype == ML::Video::WeightDType::BF16 ? weights.map { |value| video_metal_bf16_expand(value) } : weights
        effective_biases = bias_dtype == ML::Video::WeightDType::BF16 ? biases.map { |value| video_metal_bf16_expand(value) } : biases
        w = video_metal_resident_weight(arena, "#{prefix}_weight", weights, weight_dtype)
        b = video_metal_resident_weight(arena, "#{prefix}_bias", biases, bias_dtype)
        ref_w = arena.upload("#{prefix}_reference_weight", effective_weights)
        ref_b = arena.upload("#{prefix}_reference_bias", effective_biases)
        outputs = [
          arena.alloc("#{prefix}_layer_output", rows * dim),
          arena.alloc("#{prefix}_rms_output", rows * dim),
        ]
        references = [
          arena.alloc("#{prefix}_layer_reference", rows * dim),
          arena.alloc("#{prefix}_rms_reference", rows * dim),
        ]
        cases << {w, b, outputs, references, effective_weights, effective_biases}
      end

      allocations = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      {false, true}.each_with_index do |rms, mode|
        dtype_pairs.each_with_index do |(weight_dtype, bias_dtype), index|
          w, b, outputs, references, _, _ = cases[index]
          output = outputs[mode]
          reference = references[mode]
          ML::Video::MetalOps.norm(encoder, x, w, b, output, rows, dim, 1e-6_f32,
            rms: rms, affine: true, weight_dtype: weight_dtype, bias_dtype: bias_dtype)
          ML::Video::MetalOps.norm(encoder, x, arena["norm_#{index}_reference_weight"],
            arena["norm_#{index}_reference_bias"], reference, rows, dim, 1e-6_f32,
            rms: rms, affine: true)
        end
      end
      encoder.end_encoding
      command.commit
      command.wait
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)

      cases.each do |(_, _, outputs, references, effective_weights, effective_biases)|
        [false, true].each_with_index do |rms, mode|
          actual = outputs[mode].read(rows * dim)
          expected = references[mode].read(rows * dim)
          actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
          rows.times do |row|
            offset = row * dim
            mean = rms ? 0.0 : inputs[offset, dim].sum(&.to_f64) / dim
            variance = inputs[offset, dim].sum do |value|
              centered = rms ? value.to_f64 : value.to_f64 - mean
              centered * centered
            end / dim
            inverse = 1.0 / Math.sqrt(variance + 1e-6)
            dim.times do |column|
              centered = rms ? inputs[offset + column].to_f64 : inputs[offset + column].to_f64 - mean
              value = centered * inverse * effective_weights[column] + effective_biases[column]
              actual[row * dim + column].should be_close(value, 2e-5 + 2e-5 * value.abs)
            end
          end
        end
      end
    ensure
      arena.release
    end
  end

  it "rejects BF16 linear aliases and released typed weights before dispatch" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("alias_x", [1.0_f32, 2.0_f32])
      w = video_metal_resident_weight(arena, "alias_weight", [1.0_f32, 0.0_f32,
                                                              0.0_f32, 1.0_f32], ML::Video::WeightDType::BF16)
      bias = arena.upload("alias_bias", [0.0_f32, 0.0_f32])
      released = arena.alloc_weight("released_linear_weight", 4, ML::Video::WeightDType::BF16)
      released.release
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.linear(encoder, x, w, bias, x, 1, 2, 2,
          weight_dtype: ML::Video::WeightDType::BF16)
      end
      output = arena.alloc("bf16_released_output", 2)
      expect_raises(ArgumentError) do
        ML::Video::MetalOps.linear(encoder, x, released, bias, output, 1, 2, 2,
          weight_dtype: ML::Video::WeightDType::BF16)
      end
      encoder.end_encoding
      command.discard
    ensure
      arena.release
    end
  end
end
