require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_attention_compact"
require "../src/ml/video/hy_nomas_attention_mps"

private def compact_input(count : Int32, salt : Int32) : Array(Float32)
  Array(Float32).new(count) do |index|
    # Avalanche full indices to distinguish batch/head/token offsets. Setup
    # only; a short modular period would hide some address permutations.
    mixed = index.to_u32 &+ (salt.to_u32 &* 0x9e3779b9_u32)
    mixed ^= mixed >> 16
    mixed = mixed &* 0x7feb352d_u32
    mixed ^= mixed >> 15
    mixed = mixed &* 0x846ca68b_u32
    mixed ^= mixed >> 16
    ((mixed & 65535).to_i32 - 32768).to_f32 / 65536.0_f32
  end
end

private def compact_bf16_bits(value : Float32) : UInt32
  bits = value.unsafe_as(UInt32)
  ((bits.to_u64 + 0x7fff_u64 + ((bits >> 16) & 1)) & 0xffff0000_u64).to_u32
end

private def compact_compare(actual : Array(Float32), expected : Array(Float32), label : String) : Nil
  actual.size.should eq(expected.size)
  finite_violations = 0
  boundary_mismatches = 0
  raw_mismatches = 0
  max_abs = 0.0_f32
  actual.zip(expected).each do |value, reference|
    delta = (value - reference).abs
    max_abs = Math.max(max_abs, delta)
    finite_violations += 1 unless value.finite? && reference.finite? && delta <= 2e-5_f32 + 2e-5_f32 * reference.abs
    boundary_mismatches += 1 unless compact_bf16_bits(value) == compact_bf16_bits(reference)
    raw_mismatches += 1 unless value.unsafe_as(UInt32) == reference.unsafe_as(UInt32)
  end
  STDERR.puts "compact_attention #{label} n=#{actual.size} finite_violations=#{finite_violations} bf16_mismatches=#{boundary_mismatches} raw_mismatches=#{raw_mismatches} max_abs=#{max_abs}"
  finite_violations.should eq(0), label
  boundary_mismatches.should eq(0), label
end

describe ML::Video::HyNomasCompactAttentionMPSPlan do
  it "bounds compact staging and score geometry before device allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    shape = ML::Video::HyNomasCompactAttentionMPSPlan.geometry(1, 6560, 16, 128, 128, 4)
    shape.query_elements.should eq(16 * 128 * 128)
    shape.kv_elements.should eq(16 * 6560 * 128)
    shape.max_score_elements.should eq(4_i64 * 128 * 6560)
    shape.graph_calls.should eq(52 * 4)
    shape.resident_bytes.should eq(2_i64 * (shape.query_elements.to_i64 + shape.kv_elements) * 4)
    shape.resident_bytes.should be < (4_i64 * 16 * 6560 * 128 * 4)
    [{0, 9, 3, 32, 4, 2}, {1, 0, 3, 32, 4, 2},
     {1, 9, 0, 32, 4, 2}, {1, 9, 3, 0, 4, 2},
     {1, 9, 3, 32, 0, 2}, {1, 9, 3, 32, 10, 2},
     {1, 9, 3, 32, 4, 0}, {1, 9, 3, 32, 4, 4},
     {Int32::MAX, 9, 3, 32, 4, 2}].each do |invalid|
      expect_raises(ArgumentError) { ML::Video::HyNomasCompactAttentionMPSPlan.geometry(*invalid) }
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "rejects unqualified head grouping at workspace admission without allocating" do
    allocations = ML::MetalBuffer.video_spec_allocations
    expect_raises(ArgumentError, /diagnostic only/) do
      ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 257, 16, 128, 128, 4)
    end
    expect_raises(ArgumentError, /Short compact/) do
      ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 9, 16, 128, 4, 16)
    end
    expect_raises(ArgumentError, /Multi-batch compact/) do
      ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(2, 65, 16, 128, 32, 16)
    end
    ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 257, 16, 128, 128, 16)
      .graph_calls.should eq(3)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "rejects aliased and short caller bindings before staging allocation" do
    arena = ML::Video::Arena.new
    begin
      {"q", "k", "v", "output"}.each { |name| arena.alloc(name, 3 * 9 * 32) }
      arena.alloc("short", 1)
      allocations = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /alias/) do
        ML::Video::HyNomasCompactAttentionMPSPlan.new(1, 9, 3, 32,
          arena["q"], arena["k"], arena["v"], arena["q"], 4, 2)
      end
      expect_raises(ArgumentError) do
        ML::Video::HyNomasCompactAttentionMPSPlan.new(1, 9, 3, 32,
          arena["q"], arena["k"], arena["short"], arena["output"], 4, 2)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      arena.release
    end
  end

  geometries = [{2, 9, 3, 32, 9, 3}, {1, 8, 4, 32, 8, 4},
                {2, 5, 3, 32, 5, 3}, {1, 1, 1, 32, 1, 1},
                {2, 65, 3, 32, 32, 3},
                {1, 9, 16, 128, 9, 16},
                {1, 33, 16, 128, 32, 16}, {1, 65, 16, 128, 32, 16},
                {1, 65, 16, 128, 64, 16},
                {1, 129, 16, 128, 64, 16}, {1, 129, 16, 128, 128, 16},
                {1, 257, 16, 128, 128, 16}, {1, 257, 16, 128, 256, 16},
                {1, 513, 16, 128, 128, 16}, {1, 513, 16, 128, 256, 16},
                {1, 513, 16, 128, 512, 16}]
  # Retained counterexamples, not a relaxed numerical gate. The block workspace
  # rejects head grouping before staging allocation; this opt-in must still
  # fail if the recorded BF16-RNE divergence remains on the current backend.
  if ENV["HY_NOMAS_COMPACT_GROUP_DIAGNOSTIC"]? == "1"
    geometries.concat([{2, 9, 3, 32, 4, 2}, {1, 8, 4, 32, 8, 3},
                       {2, 9, 3, 32, 4, 3}, {1, 9, 16, 128, 4, 16},
                       {2, 65, 16, 128, 32, 16},
                       {1, 129, 16, 128, 129, 4}, {1, 257, 16, 128, 257, 4},
                       {1, 129, 16, 128, 64, 4}, {1, 257, 16, 128, 128, 4}])
  end
  geometries.each do |geometry|
    batches, tokens, heads, head_dim, query_chunk, heads_per_group = geometry
    input_bf16 = ENV["HY_NOMAS_COMPACT_INPUT_BF16"]? == "1"
    label = "B#{batches}/T#{tokens}/H#{heads}/D#{head_dim}/C#{query_chunk}/G#{heads_per_group}/input_bf16#{input_bf16}"
    it "matches canonical MPS with independent offsets, tails and serial reuse: #{label}" do
      arena = ML::Video::Arena.new
      compact = nil.as(ML::Video::HyNomasCompactAttentionMPSPlan?)
      canonical = nil.as(ML::Video::HyNomasAttentionMPSPlan?)
      chain = nil.as(ML::Metal::MPSCommandChain?)
      other_chain = nil.as(ML::Metal::MPSCommandChain?)
      retained_before = ML::Video::HyNomasCompactAttentionMPSPlan.retained_plan_count
      begin
        count = batches * tokens * heads * head_dim
        {"q" => 1, "k" => 2, "v" => 3}.each do |name, salt|
          original = compact_input(count, salt)
          changed = compact_input(count, salt + 7)
          if input_bf16
            original.map! { |value| compact_bf16_bits(value).unsafe_as(Float32) }
            changed.map! { |value| compact_bf16_bits(value).unsafe_as(Float32) }
          end
          arena.upload(name, original + Array(Float32).new(16, 123.0_f32))
          arena.upload("#{name}.changed", changed)
        end
        {"dense", "compact"}.each do |name|
          arena.upload(name, Array(Float32).new(count, 0.0_f32) + Array(Float32).new(16, 123.0_f32))
        end
        {"dense.first", "compact.first"}.each { |name| arena.alloc(name, count) }
        canonical = ML::Video::HyNomasAttentionMPSPlan.new(batches, tokens, heads, head_dim,
          arena["q"], arena["k"], arena["v"], arena["dense"])
        allocations = ML::MetalBuffer.video_spec_allocations
        compact = ML::Video::HyNomasCompactAttentionMPSPlan.new(batches, tokens, heads, head_dim,
          arena["q"], arena["k"], arena["v"], arena["compact"], query_chunk, heads_per_group)
        (ML::MetalBuffer.video_spec_allocations - allocations).should eq(4)
        compact.resident_bytes.should eq(compact.shape.resident_bytes)
        ML::Video::HyNomasCompactAttentionMPSPlan.retained_plan_count.should eq(retained_before + 1)
        expect_raises(ArgumentError, /binding changed/) do
          compact.validate_bindings!(arena["q.changed"], arena["k"], arena["v"], arena["compact"])
        end

        chain = ML::Metal::MPSCommandChain.new
        other_chain = ML::Metal::MPSCommandChain.new
        allocations = ML::MetalBuffer.video_spec_allocations
        canonical.encode(chain)
        compact.encode(chain)
        expect_raises(ArgumentError, /Complete/) { compact.release }
        expect_raises(ArgumentError, /in-flight/) { compact.encode(other_chain) }
        ML::Video::MetalOps.copy(chain.encoder, arena["dense"], arena["dense.first"], count)
        ML::Video::MetalOps.copy(chain.encoder, arena["compact"], arena["compact.first"], count)
        {"q", "k", "v"}.each do |name|
          ML::Video::MetalOps.copy(chain.encoder, arena["#{name}.changed"], arena[name], count)
        end
        canonical.encode(chain)
        compact.encode(chain)
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        chain.commit_and_wait
        chain.wait_count.should eq(1)
        compact_compare(arena["compact.first"].read(count), arena["dense.first"].read(count), "#{label}/first")
        compact_compare(arena["compact"].read(count), arena["dense"].read(count), "#{label}/changed")
        # A completed chain transfers ownership to another live chain without
        # reallocating or retaining a stale encoder. Each request drains once.
        allocations = ML::MetalBuffer.video_spec_allocations
        canonical.encode(other_chain)
        compact.encode(other_chain)
        expect_raises(ArgumentError, /Complete/) { compact.release }
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        other_chain.commit_and_wait
        other_chain.wait_count.should eq(1)
        compact_compare(arena["compact"].read(count), arena["dense"].read(count), "#{label}/next_chain")
        {"q", "k", "v", "dense", "compact"}.each do |name|
          arena[name].read(count + 16)[count, 16].should eq(Array(Float32).new(16, 123.0_f32)),
            "#{label}/#{name} must preserve the end canary"
        end
        compact.release
        compact.release
        expect_raises(ArgumentError, /released/) { compact.encode(other_chain) }
      ensure
        chain.try { |pending| pending.commit_and_wait unless pending.completed? }
        other_chain.try { |pending| pending.commit_and_wait unless pending.completed? }
        if (chain.nil? || chain.not_nil!.completed?) && (other_chain.nil? || other_chain.not_nil!.completed?)
          compact.try(&.release)
          canonical.try(&.release)
          chain.try(&.release)
          other_chain.try(&.release)
          arena.release
        end
      end
      ML::Video::HyNomasCompactAttentionMPSPlan.retained_plan_count.should eq(retained_before)
    end
  end
end
