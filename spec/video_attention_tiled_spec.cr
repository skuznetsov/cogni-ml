require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/metal_attention"
require "./support/video_allocation_counter"

private def tiled_attention_fixture_tensor(arena : ML::Video::Arena, name : String,
                                           record : JSON::Any, count : Int32) : ML::MetalBuffer
  path = record["path"].as_s
  bytes = count.to_i64 * 4
  record["bytes"].as_i64.should eq(bytes)
  File.size(path).should eq(bytes)
  buffer = arena.alloc(name, count)
  raw = Bytes.new(buffer.contents.as(UInt8*), bytes.to_i32)
  File.open(path, "rb") { |io| io.read_fully(raw) }
  Digest::SHA256.hexdigest(raw).should eq(record["sha256"].as_s)
  buffer
end

describe "Opt-in tiled F32 video attention" do
  it "rejects unsupported geometry, lifetimes and aliases before dispatch" do
    a = ML::Video::Arena.new
    begin
      q = a.alloc("q", 128)
      k = a.alloc("k", 128)
      v = a.alloc("v", 128)
      y = a.alloc("y", 128)
      tiny = a.alloc("tiny", 1)
      ML::Video::MetalOps.prepare_tiled_attention!
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      before = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, y, 0, 1, 1, 128) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, y, 1, 1, 1, 32) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, y, Int32::MAX, 1, Int32::MAX, 128, Int32::MAX) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled_range(e, q, k, v, y, 1, 1, 1, 128, -1, 1) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled_range(e, q, k, v, y, 1, 1, 1, 128, 0, 0) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled_range(e, q, k, v, y, 1, 1, 1, 128, 1, 1) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, q, 1, 1, 1, 128) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, k, 1, 1, 1, 128) }
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, tiny, 1, 1, 1, 128) }
      a.remove("v")
      expect_raises(ArgumentError) { ML::Video::MetalOps.attention_tiled(e, q, k, v, y, 1, 1, 1, 128) }
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      e.end_encoding
      command.discard
    ensure
      a.release
    end
  end

  it "matches every pinned MPS output with tails, batches and the full Wan key length" do
    fixture_path = ENV["VIDEO_TILED_ATTENTION_FIXTURE"]?
    pending! "Set VIDEO_TILED_ATTENTION_FIXTURE to the --long pinned MPS manifest" unless fixture_path
    fixture = JSON.parse(File.read(fixture_path))
    fixture["torch"].as_s.should eq("2.8.0")
    fixture["device"].as_s.should eq("mps")
    fixture["fallback_disabled"].as_bool.should be_true
    cases = fixture["cases"].as_a
    cases.map { |c| c["name"].as_s }.sort.should eq([
      "b1-q64-k1024-h40-d128", "b1-q64-k36608-h40-d128",
      "b1-q9-k65-h2-d64-near-tie", "b1-q9-k67-h2-d128-sharp",
      "b2-q1-k1-h1-d64-single", "b2-q13-k97-h3-d64",
    ])
    ML::Video::MetalOps.prepare_tiled_attention!
    cases.each do |c|
      a = ML::Video::Arena.new
      begin
        b, nq, nk, heads, dim = {c["batches"].as_i, c["query_rows"].as_i,
                                 c["key_rows"].as_i, c["heads"].as_i, c["dim"].as_i}
        count = b * nq * heads * dim
        q = tiled_attention_fixture_tensor(a, "q", c["q"], count)
        k = tiled_attention_fixture_tensor(a, "k", c["k"], b * nk * heads * dim)
        v = tiled_attention_fixture_tensor(a, "v", c["v"], b * nk * heads * dim)
        reference = tiled_attention_fixture_tensor(a, "expected", c["expected"], count).read(count)
        first = a.alloc("first", count)
        repeated = a.alloc("repeated", count)
        ranged = a.alloc("ranged", count)
        before = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        e = ML::Metal::ComputeEncoder.new(command)
        ML::Video::MetalOps.attention_tiled(e, q, k, v, first, nq, nk, heads, dim, b)
        ML::Video::MetalOps.attention_tiled(e, q, k, v, repeated, nq, nk, heads, dim, b)
        groups = b * ((nq + 7) // 8) * heads
        chunk_size = Math.max(groups // 3, 1)
        first_tile = 0
        while first_tile < groups
          chunk_count = Math.min(chunk_size, groups - first_tile)
          ML::Video::MetalOps.attention_tiled_range(e, q, k, v, ranged, nq, nk,
            heads, dim, first_tile, chunk_count, b)
          first_tile += chunk_count
        end
        ML::MetalBuffer.video_spec_allocations.should eq(before)
        e.end_encoding
        command.commit
        command.wait
        actual = first.read(count)
        repeated.read(count).map(&.unsafe_as(UInt32)).should eq(actual.map(&.unsafe_as(UInt32)))
        ranged.read(count).map(&.unsafe_as(UInt32)).should eq(actual.map(&.unsafe_as(UInt32)))
        violations = 0
        worst = 0_f64
        actual.each_with_index do |value, i|
          ref = reference[i]
          ratio = (value - ref).abs.to_f64 / (2e-5 + 2e-5 * ref.abs)
          violations += 1 unless value.finite? && ref.finite? && ratio <= 1
          worst = Math.max(worst, ratio)
        end
        violations.should eq(0), "#{c["name"].as_s}: violations=#{violations}, worst=#{worst}"

        # This fixture crosses query-tile, head and batch boundaries with a
        # short final query tile. Three chain roots must preserve the same bits
        # while waiting exactly once and allocating no replacement buffers.
        if c["name"].as_s == "b2-q13-k97-h3-d64"
          chained = a.alloc("chained", count)
          chain = ML::Metal::MPSCommandChain.new
          begin
            before = ML::MetalBuffer.video_spec_allocations
            ML::Video::MetalOps.attention_tiled(chain, q, k, v, chained, nq, nk,
              heads, dim, b, 5)
            ML::MetalBuffer.video_spec_allocations.should eq(before)
            chain.wait_count.should eq(0)
            chain.commit_and_wait
            chain.wait_count.should eq(1)
            chain.segment_count.should eq(3)
            chained.read(count).map(&.unsafe_as(UInt32)).should eq(actual.map(&.unsafe_as(UInt32)))
          ensure
            chain.commit_and_wait unless chain.completed?
            chain.release
          end
        end
      ensure
        a.release
      end
    end
  end
end
