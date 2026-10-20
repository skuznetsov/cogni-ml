require "spec"
require "../src/ml/video/wan_s2v_block"
require "./support/video_allocation_counter"

private def wan_rope_table_config(theta : Float32 = 10_000.0_f32) : ML::Video::WanS2VConfig
  ML::Video::WanS2VConfig.new(
    batches: 2, seq: 9, context_tokens: 3, dim: 32, heads: 2,
    ffn_dim: 64, split: 5, theta: theta,
    video_frames: 1, spatial_tokens: 2,
  )
end

private def wan_rope_values(count : Int32, salt : Int32) : Array(Float32)
  Array(Float32).new(count) do |index|
    (((index * (17 + salt * 2)) % 127) - 63).to_f32 * 0.001_f32 +
      ((index + salt) % 7).to_f32 * 0.00013_f32
  end
end

private def wan_rope_positions(cfg : ML::Video::WanS2VConfig, revision : Int32) : Array(Float32)
  Array(Float32).new(cfg.batches * cfg.seq * 3) do |index|
    axis = index % 3
    token_batch = index // 3
    token = token_batch % cfg.seq
    batch = token_batch // cfg.seq
    base = case axis
           when 0 then revision == 0 ? -17.375_f32 : 1_000_003.5_f32
           when 1 then revision == 0 ? 0.125_f32 : -65_537.75_f32
           else        revision == 0 ? 8_192.5_f32 : 3.625_f32
           end
    base + batch.to_f32 * 0.75_f32 + token.to_f32 * (axis + 1).to_f32 * 0.0625_f32
  end
end

private def wan_rope_bits(buffer : ML::MetalBuffer, count : Int32) : Array(UInt32)
  buffer.read(count).map { |value| value.to_f32.unsafe_as(UInt32) }
end

describe ML::Video::WanS2VBlock do
  it "accounts for the optional final cosine/sine table without changing the legacy plan" do
    cfg = wan_rope_table_config
    legacy = ML::Video::WanS2VBlock.scratch_plan(cfg)
    prepared = ML::Video::WanS2VBlock.scratch_plan(cfg, precompute_rope: true)
    table_values = cfg.batches.to_i64 * cfg.seq * cfg.head_dim

    (prepared.sum { |_, count| count.to_i64 } - legacy.sum { |_, count| count.to_i64 }).should eq(table_values)
    (ML::Video::WanS2VBlock.scratch_bytes(cfg, precompute_rope: true) -
      ML::Video::WanS2VBlock.scratch_bytes(cfg)).should eq(table_values * sizeof(Float32))
    cfg.rope_axis_dims.should eq({4, 2, 2})
  end

  it "matches inline rotations bitwise, refreshes changed positions, and keeps encode allocation-free" do
    cfg = wan_rope_table_config
    ML::Video::WanS2VBlock.prepare!(precompute_rope: true)

    inputs = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    begin
      ML::Video::WanS2VBlock.prepare_rope_scratch!(scratch, cfg, precompute_rope: true)
      q_values = wan_rope_values(cfg.values, 1)
      k_values = wan_rope_values(cfg.values, 4)
      q_inline = inputs.upload("q_inline", q_values.dup)
      k_inline = inputs.upload("k_inline", k_values.dup)
      q_table = inputs.upload("q_table", q_values.dup)
      k_table = inputs.upload("k_table", k_values.dup)
      positions = inputs.upload("positions", wan_rope_positions(cfg, 0))
      q_inline_out = outputs.alloc("q_inline_out", cfg.values)
      k_inline_out = outputs.alloc("k_inline_out", cfg.values)
      q_table_out = outputs.alloc("q_table_out", cfg.values)
      k_table_out = outputs.alloc("k_table_out", cfg.values)

      2.times do |revision|
        positions.write(wan_rope_positions(cfg, revision))
        q_now = wan_rope_values(cfg.values, revision + 2)
        k_now = wan_rope_values(cfg.values, revision + 5)
        q_inline.write(q_now)
        q_table.write(q_now)
        k_inline.write(k_now)
        k_table.write(k_now)

        chain = ML::Metal::MPSCommandChain.new
        begin
          allocation_start = ML::MetalBuffer.video_spec_allocations
          encoder = chain.encoder
          ML::Video::WanS2VBlock.encode_rope_pair(
            encoder, q_inline, k_inline, positions, q_inline_out, k_inline_out,
            cfg, scratch, precompute_rope: false)
          ML::Video::WanS2VBlock.encode_rope_pair(
            encoder, q_table, k_table, positions, q_table_out, k_table_out,
            cfg, scratch, precompute_rope: true)
          ML::MetalBuffer.video_spec_allocations.should eq(allocation_start),
            "RoPE encoding must not allocate temporary Metal buffers"
          chain.wait_count.should eq(0), "the pair helper must not introduce a command wait"
          chain.commit_and_wait
          chain.wait_count.should eq(1)

          wan_rope_bits(q_inline_out, cfg.values).should eq(wan_rope_bits(q_table_out, cfg.values))
          wan_rope_bits(k_inline_out, cfg.values).should eq(wan_rope_bits(k_table_out, cfg.values))
        ensure
          chain.commit_and_wait unless chain.completed?
          chain.release if chain.completed?
        end
      end

      # In-place Q/K output must retain the same bitwise contract.
      q_inline_inplace = inputs.upload("q_inline_inplace", wan_rope_values(cfg.values, 7))
      k_inline_inplace = inputs.upload("k_inline_inplace", wan_rope_values(cfg.values, 9))
      q_table_inplace = inputs.upload("q_table_inplace", wan_rope_values(cfg.values, 7))
      k_table_inplace = inputs.upload("k_table_inplace", wan_rope_values(cfg.values, 9))
      chain = ML::Metal::MPSCommandChain.new
      begin
        allocation_start = ML::MetalBuffer.video_spec_allocations
        encoder = chain.encoder
        ML::Video::WanS2VBlock.encode_rope_pair(
          encoder, q_inline_inplace, k_inline_inplace, positions,
          q_inline_inplace, k_inline_inplace, cfg, scratch,
        )
        ML::Video::WanS2VBlock.encode_rope_pair(
          encoder, q_table_inplace, k_table_inplace, positions,
          q_table_inplace, k_table_inplace, cfg, scratch, precompute_rope: true)
        ML::MetalBuffer.video_spec_allocations.should eq(allocation_start)
        chain.wait_count.should eq(0)
        chain.commit_and_wait
        chain.wait_count.should eq(1)
        wan_rope_bits(q_inline_inplace, cfg.values).should eq(wan_rope_bits(q_table_inplace, cfg.values))
        wan_rope_bits(k_inline_inplace, cfg.values).should eq(wan_rope_bits(k_table_inplace, cfg.values))
      ensure
        chain.commit_and_wait unless chain.completed?
        chain.release if chain.completed?
      end
    ensure
      scratch.release
      outputs.release
      inputs.release
    end
  end

  it "rejects a config-key mismatch and malformed or aliased coefficient tables before dispatch" do
    cfg = wan_rope_table_config
    other_cfg = wan_rope_table_config(10_001.0_f32)
    ML::Video::WanS2VBlock.prepare!(precompute_rope: true)
    inputs = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    malformed = ML::Video::Arena.new
    begin
      ML::Video::WanS2VBlock.prepare_rope_scratch!(scratch, cfg, precompute_rope: true)
      q = inputs.upload("q", wan_rope_values(cfg.values, 2))
      k = inputs.upload("k", wan_rope_values(cfg.values, 3))
      positions = inputs.upload("positions", wan_rope_positions(cfg, 0))
      q_out = inputs.alloc("q_out", cfg.values)
      k_out = inputs.alloc("k_out", cfg.values)

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode_rope_pair(
          encoder, q, k, positions, q_out, k_out, other_cfg, scratch,
          precompute_rope: true,
        )
      end
      encoder.end_encoding
      command.discard

      table_name = scratch.buffers.keys.find { |name| name.includes?("rope_cos") } ||
                   raise "precomputed cosine table was not discoverable in its Arena"
      original_table = scratch[table_name]
      short_table = malformed.alloc("short_table", 1)
      scratch.buffers[table_name] = short_table
      begin
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        expect_raises(ArgumentError) do
          ML::Video::WanS2VBlock.encode_rope_pair(
            encoder, q, k, positions, q_out, k_out, cfg, scratch,
            precompute_rope: true,
          )
        end
        encoder.end_encoding
        command.discard
      ensure
        scratch.buffers[table_name] = original_table
      end

      scratch.buffers[table_name] = q
      begin
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        expect_raises(ArgumentError) do
          ML::Video::WanS2VBlock.encode_rope_pair(
            encoder, q, k, positions, q_out, k_out, cfg, scratch,
            precompute_rope: true,
          )
        end
        encoder.end_encoding
        command.discard
      ensure
        scratch.buffers[table_name] = original_table
      end
    ensure
      malformed.release
      scratch.release
      inputs.release
    end
  end
end
