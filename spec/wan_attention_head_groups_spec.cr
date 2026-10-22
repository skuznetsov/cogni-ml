require "spec"
require "../src/ml/video/metal_attention_mps"

describe "Independent-head MPS workspace admission" do
  it "rejects invalid grouping, missing prepared K/V, aliases and small scratch before pipeline setup" do
    arena = ML::Video::Arena.new
    begin
      q = arena.alloc("group_guard_q", 256)
      k = arena.alloc("group_guard_k", 256)
      v = arena.alloc("group_guard_v", 256)
      output = arena.alloc("group_guard_output", 256)
      cq = arena.alloc("group_guard_cq", 128)
      ca = arena.alloc("group_guard_ca", 128)
      cb = arena.alloc("group_guard_cb", 128)
      pk = arena.alloc("group_guard_pk", 256)
      pv = arena.alloc("group_guard_pv", 256)
      tiny = arena.alloc("group_guard_tiny", 1)
      ML::Metal::PipelineCache.clear
      {0, -1, 3, 5}.each do |group|
        expect_raises(ArgumentError) do
          ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 4, 64, 1,
            q, k, v, output, cq, ca, cb, pk, pv, heads_per_group: group)
        end
      end
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 4, 64, 1,
          q, k, v, output, cq, ca, cb, heads_per_group: 2)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 4, 64, 1,
          q, k, v, output, cq, ca, cb, pk, pv,
          heads_per_group: 2, explicit_qk_softmax_pv: true)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 4, 64, 1,
          q, k, v, output, cq, ca, cb, k, pv, heads_per_group: 2)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 4, 64, 1,
          q, k, v, output, tiny, ca, cb, pk, pv, heads_per_group: 2)
      end
      ML::Metal::PipelineCache.entry_count.should eq(0),
        "Invalid head grouping prepared GPU pipelines before rejection"
    ensure
      arena.release
    end
  end
end
