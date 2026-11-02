require "spec"
require "../src/ml/video/hy_nomas_reference_block_mps"
require "./support/video_allocation_counter"

# Fail the borrowed MPS operation after pause! without adding a production hook.
class ML::Video::HyNomasMPSRMS::Plan
  def reference_spec_release_mean! : Nil
    @mean_plan.not_nil!.release
  end
end

class ML::Video::HyNomasReferenceMPSWorkspace
  def reference_spec_release_img_q_mean! : Nil
    @rms_plans["img.q"].reference_spec_release_mean!
  end
end

HY_NOMAS_REFERENCE_WORKSPACE_CACHE_MUTATION = <<-METAL
#include <metal_stdlib>
using namespace metal;
kernel void hy_nomas_reference_workspace_cache_mutation(
    device ushort *key [[buffer(0)]], device ushort *value [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id < count) {
    key[id] = 0x3f80; // BF16 1.0
    value[id] = 0xc000; // BF16 -2.0
  }
}
kernel void hy_nomas_reference_workspace_poison_prospective(
    device float *key [[buffer(0)]], device float *value [[buffer(1)]],
    device float *context [[buffer(2)]], device float *query [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
  if (id < count) {
    key[id] = 123.0f;
    value[id] = 321.0f;
    context[id] = 9.0f;
    query[id] = 0.0f;
  }
}
kernel void hy_nomas_reference_workspace_seed_attention(
    device ushort *key [[buffer(0)]], device ushort *value [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &headDim [[buffer(4)]], uint id [[thread_position_in_grid]]) {
  if (id < count) {
    uint rowWidth = heads * headDim;
    uint token = id / rowWidth;
    uint tag = id % rowWidth + 1;
    key[id] = 0; // BF16 zero makes QK logits exactly zero.
    // Distinct BF16 values per token/head/channel. Their two-token mean is
    // 1.25 + tag/128, also exactly representable in BF16.
    value[id] = ushort((token == 0 ? 0x3f80 : 0x3fc0) + tag);
  }
}
METAL

class HyNomasReferenceSpecHistoryProvider < ML::Video::HyNomasMPSHistoryProvider
  def resident_bytes : Int64
    0_i64
  end

  def validate_bindings!(tokens : Int32, heads : Int32, head_dim : Int32,
                         query : ML::MetalBuffer, key : ML::MetalBuffer,
                         value : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
  end

  def validate_chain!(chain : ML::Metal::MPSCommandChain) : Nil
  end

  def encode(chain : ML::Metal::MPSCommandChain, block_index : Int32) : Nil
  end

  def release : Nil
  end
end

private def hy_nomas_reference_workspace_config(
  *, batches : Int32 = 1, video_tokens : Int32 = 1, text_tokens : Int32 = 1,
  img_modulation_rows : Int32 = 0, txt_modulation_rows : Int32 = 0,
  precision : ML::Video::HyNomasAttentionPrecision = ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl,
) : ML::Video::HyNomasConfig
  ML::Video::HyNomasConfig.new(batches: batches, video_tokens: video_tokens,
    text_tokens: text_tokens, dim: 64, heads: 2, head_dim: 32, ffn_dim: 256,
    img_modulation_rows: img_modulation_rows,
    txt_modulation_rows: txt_modulation_rows,
    attention_precision: precision)
end

private def hy_nomas_reference_workspace_weights(
  config : ML::Video::HyNomasConfig,
) : ML::Video::Arena
  arena = ML::Video::Arena.new
  ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config).each do |binding|
    count = binding.shape.reduce(1_i64) { |product, dim| product * dim }.to_i32
    buffer = arena.alloc_weight(binding.target, count, ML::Video::WeightDType::BF16)
    bytes = Bytes.new(count * 2, 0_u8)
    if binding.target.ends_with?("_norm_weight")
      count.times do |index|
        bytes[index * 2] = 0x80_u8
        bytes[index * 2 + 1] = 0x3f_u8
      end
    end
    buffer.write_bytes(bytes.to_unsafe, bytes.size)
  end
  arena
end

private def hy_nomas_reference_workspace_inputs(
  arena : ML::Video::Arena, config : ML::Video::HyNomasConfig,
) : Nil
  arena.upload("img", Array(Float32).new(config.image_rows * config.dim, 0.125_f32))
  arena.upload("txt", Array(Float32).new(config.text_rows * config.dim, -0.125_f32))
  arena.upload("vec", Array(Float32).new(config.effective_img_modulation_rows * config.dim, 0.0_f32))
  arena.upload("vec_txt", Array(Float32).new(config.effective_txt_modulation_rows * config.dim, 0.0_f32))
  arena.upload("rope_cos", Array(Float32).new(config.video_tokens * config.head_dim, 1.0_f32))
  arena.upload("rope_sin", Array(Float32).new(config.video_tokens * config.head_dim, 0.0_f32))
end

describe ML::Video::HyNomasReferenceMPSWorkspace do
  it "attends to physical BF16 history after prospective K/V are poisoned and retains the latest consumer lease" do
    config = hy_nomas_reference_workspace_config(text_tokens: 2)
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    arenas = [] of ML::Video::Arena
    workspace = nil.as(ML::Video::HyNomasReferenceMPSWorkspace?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    other_chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      weights = hy_nomas_reference_workspace_weights(config)
      arenas << weights
      scratch = ML::Video::Arena.new
      arenas << scratch
      ML::Video::HyNomasDualStreamBlock.scratch_plan(config).each { |name, count| scratch.alloc(name, count) }
      block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
      inputs = ML::Video::Arena.new
      arenas << inputs
      hy_nomas_reference_workspace_inputs(inputs, config)
      poison = ML::Metal::ComputePipeline.new("hy_nomas_reference_workspace_poison_prospective", HY_NOMAS_REFERENCE_WORKSPACE_CACHE_MUTATION)
      seed = ML::Metal::ComputePipeline.new("hy_nomas_reference_workspace_seed_attention", HY_NOMAS_REFERENCE_WORKSPACE_CACHE_MUTATION)
      workspace = ML::Video::HyNomasReferenceMPSWorkspace.new(block,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"], valid_text_tokens: 1,
        reference_history: true, reference_attention: true)
      cache = workspace.not_nil!.reference_history_cache
      attention = workspace.not_nil!.reference_attention_plan
      attention.resident_bytes.should eq(512_i64)
      chain = ML::Metal::MPSCommandChain.new
      expect_raises(ArgumentError, /must be stored before read/) { workspace.not_nil!.encode_reference_attention(chain.not_nil!) }
      allocations_before = ML::MetalBuffer.video_spec_allocations
      block.encode_mps_reference_history(chain.not_nil!, inputs["img"], inputs["txt"],
        inputs["vec"], inputs["vec_txt"], inputs["rope_cos"], inputs["rope_sin"], workspace.not_nil!)
      chain.not_nil!.encoder.set_pipeline(poison)
        .set_buffer(block.checkpoints["joint.key"], 0, ML::Metal::BufferAccess::Write)
        .set_buffer(block.checkpoints["joint.value"], 1, ML::Metal::BufferAccess::Write)
        .set_buffer(block.checkpoints["joint.attention"], 2, ML::Metal::BufferAccess::Write)
        .set_buffer(block.checkpoints["joint.query"], 3, ML::Metal::BufferAccess::Write)
        .set_value((config.joint_tokens * config.dim).to_u32, 4)
        .dispatch_1d(config.joint_tokens * config.dim)
      chain.not_nil!.encoder.set_pipeline(seed)
        .set_buffer(cache.key_storage, 0, ML::Metal::BufferAccess::Write)
        .set_buffer(cache.value_storage, 1, ML::Metal::BufferAccess::Write)
        .set_value(cache.geometry.cache_elements.to_u32, 2)
        .set_value(cache.heads.to_u32, 3)
        .set_value(cache.head_dim.to_u32, 4)
        .dispatch_1d(cache.geometry.cache_elements)
      context = workspace.not_nil!.encode_reference_attention(chain.not_nil!)
      context.same?(block.checkpoints["joint.attention"]).should be_true
      # A different in-flight chain cannot borrow these same bindings.
      other_chain = ML::Metal::MPSCommandChain.new
      expect_raises(ArgumentError, /in-flight command chains/) { workspace.not_nil!.encode_reference_attention(other_chain.not_nil!) }
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before)
      chain.not_nil!.wait_count.should eq(0)
      expect_raises(ArgumentError, /Drain.*before release/) { attention.release }
      expect_raises(ArgumentError, /Complete reference BF16 chain/) { workspace.not_nil!.release }
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      values = context.read(config.joint_tokens * config.dim)
      active_row_width = config.heads * config.head_dim
      expected = Array(Float32).new(cache.geometry.cache_elements) do |index|
        1.25_f32 + (index % active_row_width + 1).to_f32 / 128.0_f32
      end
      values.first(cache.geometry.cache_elements).should eq(expected)
      values.skip(cache.geometry.cache_elements).should eq(Array.new(config.dim, 9.0_f32))
      chain.not_nil!.release
      chain = ML::Metal::MPSCommandChain.new
      workspace.not_nil!.encode_reference_attention(chain.not_nil!)
      expect_raises(ArgumentError, /Drain.*before release/) { cache.release }
      expect_raises(ArgumentError, /Drain.*before release/) { workspace.not_nil!.release }
      attention.query_bf16.valid?.should be_true
      cache.key_storage.valid?.should be_true
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)
      chain.not_nil!.commit_and_wait
      workspace.not_nil!.release
      workspace = nil
      attention.resident_bytes.should eq(0_i64)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    ensure
      {chain, other_chain}.each do |pending|
        if pending
          pending.commit_and_wait unless pending.completed?
        end
      end
      workspace.try(&.release)
      chain.try(&.release)
      other_chain.try(&.release)
      arenas.reverse_each(&.release)
    end
  end

  it "owns an opt-in physical history cache and preflights its single-write lease" do
    config = hy_nomas_reference_workspace_config(text_tokens: 2)
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    arenas = [] of ML::Video::Arena
    workspace = nil.as(ML::Video::HyNomasReferenceMPSWorkspace?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      weights = hy_nomas_reference_workspace_weights(config)
      arenas << weights
      scratch = ML::Video::Arena.new
      arenas << scratch
      ML::Video::HyNomasDualStreamBlock.scratch_plan(config).each do |name, count|
        scratch.alloc(name, count)
      end
      block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
      inputs = ML::Video::Arena.new
      arenas << inputs
      hy_nomas_reference_workspace_inputs(inputs, config)
      mutation = ML::Metal::ComputePipeline.new(
        "hy_nomas_reference_workspace_cache_mutation",
        HY_NOMAS_REFERENCE_WORKSPACE_CACHE_MUTATION)
      workspace = ML::Video::HyNomasReferenceMPSWorkspace.new(block,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"], valid_text_tokens: 1,
        reference_history: true)
      cache = workspace.not_nil!.reference_history_cache
      cache.shape.should eq({2, 2, 32})
      cache.resident_bytes.should eq(1536_i64)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)
      chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.video_spec_allocations
      key, value = block.encode_mps_reference_history(chain.not_nil!,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"], workspace.not_nil!)
      key.same?(cache.key_read).should be_true
      value.same?(cache.value_read).should be_true
      key.same?(block.checkpoints["joint.key"]).should be_false
      expect_raises(ArgumentError, /single-write/) do
        block.encode_mps_reference_history(chain.not_nil!,
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
          inputs["rope_cos"], inputs["rope_sin"], workspace.not_nil!)
      end
      # A subsequent read must consume physical BF16 bytes, even when those
      # differ from the current block's all-zero prospective K/V.
      chain.not_nil!.encoder.set_pipeline(mutation)
        .set_buffer(cache.key_storage, 0, ML::Metal::BufferAccess::Write)
        .set_buffer(cache.value_storage, 1, ML::Metal::BufferAccess::Write)
        .set_value(cache.geometry.cache_elements.to_u32, 2)
        .dispatch_1d(cache.geometry.cache_elements)
      cache.read(chain.not_nil!)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before)
      chain.not_nil!.wait_count.should eq(0)
      expect_raises(ArgumentError, /Complete reference BF16 chain/) { workspace.not_nil!.release }
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      key.read(cache.geometry.cache_elements).should eq(Array.new(cache.geometry.cache_elements, 1.0_f32))
      value.read(cache.geometry.cache_elements).should eq(Array.new(cache.geometry.cache_elements, -2.0_f32))
      block.checkpoints["joint.key"].read(config.joint_tokens * config.dim).should eq(Array.new(config.joint_tokens * config.dim, 0.0_f32))
      chain.not_nil!.release
      chain = ML::Metal::MPSCommandChain.new
      cache.read(chain.not_nil!)
      # The workspace's original chain has drained, but the owned cache has a
      # newer read lease. Its release guard must prevent partial teardown.
      expect_raises(ArgumentError, /Drain.*before release/) { workspace.not_nil!.release }
      cache.key_read.valid?.should be_true
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)
      chain.not_nil!.commit_and_wait
      workspace.not_nil!.release
      workspace = nil
      cache.resident_bytes.should eq(0_i64)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    ensure
      if pending = chain
        pending.commit_and_wait unless pending.completed?
      end
      workspace.try(&.release)
      chain.try(&.release)
      arenas.reverse_each(&.release)
    end
  end

  it "consumes the history attempt after partial preattention failure but preserves preflight rejection" do
    config = hy_nomas_reference_workspace_config
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    arenas = [] of ML::Video::Arena
    workspace = nil.as(ML::Video::HyNomasReferenceMPSWorkspace?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      weights = hy_nomas_reference_workspace_weights(config)
      arenas << weights
      scratch = ML::Video::Arena.new
      arenas << scratch
      ML::Video::HyNomasDualStreamBlock.scratch_plan(config).each do |name, count|
        scratch.alloc(name, count)
      end
      block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
      inputs = ML::Video::Arena.new
      arenas << inputs
      hy_nomas_reference_workspace_inputs(inputs, config)
      wrong_img = inputs.upload("wrong_img", Array.new(config.image_rows * config.dim, 0.125_f32))
      workspace = ML::Video::HyNomasReferenceMPSWorkspace.new(block,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"], reference_history: true)
      workspace.not_nil!.reference_spec_release_img_q_mean!
      chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /input identity changed/) do
        block.encode_mps_reference_history(chain.not_nil!, wrong_img,
          inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["rope_cos"],
          inputs["rope_sin"], workspace.not_nil!)
      end
      expect_raises(Exception, /RMS mean plan has been released/) do
        block.encode_mps_reference_history(chain.not_nil!, inputs["img"],
          inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["rope_cos"],
          inputs["rope_sin"], workspace.not_nil!)
      end
      expect_raises(ArgumentError, /history attempt already started/) do
        block.encode_mps_reference_history(chain.not_nil!, inputs["img"],
          inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["rope_cos"],
          inputs["rope_sin"], workspace.not_nil!)
      end
      expect_raises(ArgumentError, /requires encode_mps_reference_history/) do
        block.encode_mps_preattention(chain.not_nil!, inputs["img"],
          inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["rope_cos"],
          inputs["rope_sin"], workspace.not_nil!)
      end
      chain.not_nil!.encoder
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before)
      chain.not_nil!.wait_count.should eq(0)
      expect_raises(ArgumentError, /Complete reference BF16 chain/) { workspace.not_nil!.release }
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
    ensure
      if pending = chain
        pending.commit_and_wait unless pending.completed?
      end
      workspace.try(&.release)
      chain.try(&.release)
      arenas.reverse_each(&.release)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    end
  end

  it "restores the encoder after a source-BF16 RMS graph failure" do
    config = hy_nomas_reference_workspace_config
    arena = ML::Video::Arena.new
    plan = nil.as(ML::Video::HyNomasMPSRMS::Plan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      input = arena.upload("input", Array(Float32).new(config.dim, 0.125_f32))
      output = arena.alloc("output", config.dim)
      gain = arena.alloc_weight("gain", config.head_dim, ML::Video::WeightDType::BF16)
      gain_bytes = Bytes.new(config.head_dim * 2, 0_u8)
      config.head_dim.times do |index|
        gain_bytes[index * 2] = 0x80_u8
        gain_bytes[index * 2 + 1] = 0x3f_u8
      end
      gain.write_bytes(gain_bytes.to_unsafe, gain_bytes.size)
      plan = ML::Video::HyNomasMPSRMS::Plan.new(config, input, gain, output,
        ML::Video::WeightDType::BF16, config.video_tokens, source_bf16: true)
      plan.not_nil!.reference_spec_release_mean!
      chain = ML::Metal::MPSCommandChain.new

      allocations_before = ML::MetalBuffer.video_spec_allocations
      expect_raises(Exception, /RMS mean plan has been released/) do
        plan.not_nil!.encode(chain.not_nil!)
      end
      # A live encoder is available again, with queued work still undrained.
      chain.not_nil!.encoder
      chain.not_nil!.wait_count.should eq(0)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before)
      expect_raises(ArgumentError, /Complete Hy-Nomas MPS RMS command chain/) do
        plan.not_nil!.release
      end
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      plan.not_nil!.release
      plan = nil
    ensure
      if active_chain = chain
        active_chain.commit_and_wait unless active_chain.completed?
      end
      plan.try(&.release)
      chain.try(&.release)
      arena.release
    end
  end

  it "admits only batch-one global BF16-boundary geometry and accounts its extra bytes" do
    valid = hy_nomas_reference_workspace_config
    ML::Video::HyNomasReferenceMPSWorkspace.validate_reference_config!(valid)
    ML::Video::HyNomasReferenceMPSWorkspace.reference_extra_bytes(valid).should eq(3328_i64)

    batch_two = hy_nomas_reference_workspace_config(batches: 2)
    expect_raises(ArgumentError, /batch-one global modulation/) do
      ML::Video::HyNomasReferenceMPSWorkspace.validate_reference_config!(batch_two)
    end

    per_token_modulation = hy_nomas_reference_workspace_config(
      video_tokens: 2, img_modulation_rows: 2)
    expect_raises(ArgumentError, /batch-one global modulation/) do
      ML::Video::HyNomasReferenceMPSWorkspace.validate_reference_config!(per_token_modulation)
    end

    f32_control = hy_nomas_reference_workspace_config(
      precision: ML::Video::HyNomasAttentionPrecision::F32Control)
    expect_raises(ArgumentError, /BF16 boundary packing/) do
      ML::Video::HyNomasReferenceMPSWorkspace.validate_reference_config!(f32_control)
    end
  end

  it "keeps ownership until the terminal drain and stops at prospective K/V" do
    config = hy_nomas_reference_workspace_config
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    arenas = [] of ML::Video::Arena
    workspace = nil.as(ML::Video::HyNomasReferenceMPSWorkspace?)
    base_workspace = nil.as(ML::Video::HyNomasMPSWorkspace?)
    chain = nil.as(ML::Metal::MPSCommandChain?)

    begin
      weights = hy_nomas_reference_workspace_weights(config)
      arenas << weights
      scratch = ML::Video::Arena.new
      arenas << scratch
      ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
      block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
      inputs = ML::Video::Arena.new
      arenas << inputs
      hy_nomas_reference_workspace_inputs(inputs, config)

      # Invalid source-BF16 RMS setup fails after the base plans have allocated
      # their scratch; all of those temporary buffers and lifetime roots must
      # be released by the constructor's rescue path.
      bytes_before_bad_setup = ML::MetalBuffer.stats[:live_bytes]
      expect_raises(ArgumentError, /requires reduction tokens to match active tokens/) do
        ML::Video::HyNomasReferenceMPSWorkspace.new(block,
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
          inputs["rope_cos"], inputs["rope_sin"], text_rms_tokens: 2)
      end
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
      ML::MetalBuffer.stats[:live_bytes].should eq(bytes_before_bad_setup)

      workspace = ML::Video::HyNomasReferenceMPSWorkspace.new(block,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"])
      workspace.reference_preattention_bf16.should be_true
      expect_raises(ArgumentError, /history cache was not prepared/) do
        workspace.reference_history_cache
      end
      expected_extra = ML::Video::HyNomasReferenceMPSWorkspace.reference_extra_bytes(config)
      base_workspace = ML::Video::HyNomasMPSWorkspace.new(block,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"])
      (workspace.resident_bytes - base_workspace.not_nil!.resident_bytes).should eq(expected_extra)
      base_workspace.not_nil!.release
      base_workspace = nil

      chain = ML::Metal::MPSCommandChain.new
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)
      ML::Video::HyNomasReferenceMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)

      allocations_before_rejected_full_encode = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /requires encode_mps_preattention/) do
        block.encode_mps(chain.not_nil!, inputs["img"], inputs["txt"],
          inputs["vec"], inputs["vec_txt"], inputs["rope_cos"],
          inputs["rope_sin"], workspace.not_nil!)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_rejected_full_encode)
      chain.not_nil!.wait_count.should eq(0)

      expect_raises(ArgumentError, /does not admit serial blocks/) do
        workspace.not_nil!.register_serial_blocks!(
          [] of ML::Video::HyNomasDualStreamBlock,
          [] of Tuple(ML::MetalBuffer, ML::MetalBuffer))
      end
      expect_raises(ArgumentError, /does not admit history attachment/) do
        workspace.not_nil!.attach_history_provider!(HyNomasReferenceSpecHistoryProvider.new)
      end

      allocations_before_encode = ML::MetalBuffer.video_spec_allocations
      key, value = block.encode_mps_preattention(chain.not_nil!,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["rope_cos"], inputs["rope_sin"], workspace.not_nil!)
      key.same?(block.checkpoints["joint.key"]).should be_true
      value.same?(block.checkpoints["joint.value"]).should be_true
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_encode)
      chain.not_nil!.wait_count.should eq(0)

      expect_raises(ArgumentError, /Complete reference BF16 chain/) do
        workspace.not_nil!.release
      end
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)

      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      workspace.not_nil!.release
      workspace = nil
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
      chain.not_nil!.release
      chain = nil
    ensure
      if active_chain = chain
        active_chain.commit_and_wait unless active_chain.completed?
        workspace.try(&.release)
        active_chain.release
      else
        workspace.try(&.release)
      end
      base_workspace.try(&.release)
      arenas.reverse_each(&.release)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    end
  end
end
