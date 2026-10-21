require "spec"
require "../src/ml/video/wan_s2v_block"
require "./support/video_allocation_counter"

private def wan_kv_preparation_config(heads : Int32, seq : Int32,
                                      head_dim : Int32 = 16) : ML::Video::WanS2VConfig
  ML::Video::WanS2VConfig.new(
    batches: 1, seq: seq, context_tokens: 3, dim: heads * head_dim, heads: heads,
    ffn_dim: heads * head_dim * 2, split: seq // 2, theta: 10_000.0_f32,
    video_frames: 1, spatial_tokens: 2,
  )
end

private def wan_kv_preparation_values(count : Int32, salt : Int32) : Array(Float32)
  Array(Float32).new(count) do |index|
    (((index * (17 + salt * 2)) % 127) - 63).to_f32 * 0.001_f32 +
      ((index + salt) % 7).to_f32 * 0.00013_f32
  end
end

private def wan_kv_preparation_bias(dim : Int32, revision : Int32) : Array(Float32)
  Array(Float32).new(dim) do |index|
    sign = index.even? ? -1.0_f32 : 1.0_f32
    sign * (0.004_f32 + ((index * 7 + revision * 3) % 19).to_f32 * 0.0007_f32)
  end
end

private def wan_kv_preparation_positions(cfg : ML::Video::WanS2VConfig,
                                         revision : Int32) : Array(Float32)
  Array(Float32).new(cfg.rows * 3) do |index|
    axis = index % 3
    token = index // 3
    base = case axis
           when 0 then revision.even? ? -17.375_f32 : 1_000_003.5_f32
           when 1 then revision.even? ? 0.125_f32 : -65_537.75_f32
           else        revision.even? ? 8_192.5_f32 : 3.625_f32
           end
    base + token.to_f32 * (axis + 1).to_f32 * 0.0625_f32
  end
end

private def wan_kv_preparation_bf16(value : Float32) : Float32
  (value.unsafe_as(UInt32) & 0xffff0000_u32).unsafe_as(Float32)
end

private def wan_kv_preparation_write_bias(buffer : ML::MetalBuffer,
                                          values : Array(Float32),
                                          dtype : ML::Video::WeightDType) : Nil
  case dtype
  when ML::Video::WeightDType::F32
    buffer.write(values)
  when ML::Video::WeightDType::BF16
    bytes = Bytes.new(values.size * sizeof(UInt16))
    values.each_with_index do |value, index|
      bits = (value.unsafe_as(UInt32) >> 16).to_u16
      bytes[index * 2] = (bits & 0xff).to_u8
      bytes[index * 2 + 1] = (bits >> 8).to_u8
    end
    buffer.write_bytes(bytes.to_unsafe, bytes.size)
  end
end

private def wan_kv_preparation_bits(buffer : ML::MetalBuffer,
                                    count : Int32) : Array(UInt32)
  buffer.read(count).map { |value| value.unsafe_as(UInt32) }
end

private def wan_kv_preparation_assert_finite(buffer : ML::MetalBuffer,
                                             count : Int32,
                                             label : String) : Nil
  values = buffer.read(count)
  values.each_with_index do |value, index|
    value.finite?.should(be_true, "#{label}[#{index}] must be finite")
  end
end

describe "Wan S2V fused K RoPE and V bias preparation" do
  it "matches separate RoPE, source-dtype V bias, and head-major packing across shapes and refreshed inputs" do
    ML::Video::WanS2VBlock.prepare!
    ML::Video::MetalAttentionMPS.prepare!
    ML::Video::MetalAttentionMPS.prepare_head_major_kv!

    { {2, 9, 16}, {3, 11, 16}, {2, 9, 128} }.each do |heads, seq, head_dim|
      cfg = wan_kv_preparation_config(heads, seq, head_dim)
      cfg.batches.should eq(1)
      cfg.head_dim.should eq(head_dim)
      pairs = head_dim // 2
      axis_thirds = pairs // 3
      cfg.rope_axis_dims.should eq({pairs - 2 * axis_thirds, axis_thirds, axis_thirds})

      bias_dtypes = {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16}
      bias_dtypes.each do |bias_dtype|
        inputs = ML::Video::Arena.new
        rope = ML::Video::Arena.new
        reference = ML::Video::Arena.new
        candidate = ML::Video::Arena.new
        begin
          ML::Video::WanS2VBlock.prepare_rope_scratch!(rope, cfg)
          frequency_hi_name = rope.buffers.keys.find { |name| name.includes?("rope_freq_hi") } ||
                              raise("missing RoPE hi frequencies")
          frequency_lo_name = rope.buffers.keys.find { |name| name.includes?("rope_freq_lo") } ||
                              raise("missing RoPE lo frequencies")
          frequency_hi = rope[frequency_hi_name]
          frequency_lo = rope[frequency_lo_name]
          k_norm = inputs.upload("k_norm", wan_kv_preparation_values(cfg.values, 1))
          v_unbiased = inputs.upload("v_unbiased", wan_kv_preparation_values(cfg.values, 4))
          v_bias = inputs.alloc_weight("v_bias", cfg.dim, bias_dtype)
          positions = inputs.upload("positions", wan_kv_preparation_positions(cfg, 0))
          q_dummy = inputs.upload("q_dummy", Array(Float32).new(cfg.values, 0.0_f32))

          # The oracle uses independent source copies so trace=true can be
          # checked without changing the reference inputs.
          k_reference = reference.upload("k_reference", wan_kv_preparation_values(cfg.values, 1))
          v_reference = reference.upload("v_reference", wan_kv_preparation_values(cfg.values, 4))
          q_rotated = reference.alloc("q_rotated", cfg.values)
          k_rotated = reference.alloc("k_rotated", cfg.values)
          expanded_bias = reference.upload("expanded_bias", Array(Float32).new(cfg.values, 0.0_f32))
          v_biased = reference.alloc("v_biased", cfg.values)
          reference_k = reference.alloc("reference_k", cfg.values)
          reference_v = reference.alloc("reference_v", cfg.values)
          prepared_k = candidate.alloc("prepared_k", cfg.values)
          prepared_v = candidate.alloc("prepared_v", cfg.values)

          trace_modes = { {false, false}, {true, false}, {false, true}, {true, true} }
          trace_modes.each_with_index do |(trace_k, trace_v), revision|
            k_values = wan_kv_preparation_values(cfg.values, revision + 1)
            v_values = wan_kv_preparation_values(cfg.values, revision + 4)
            bias_values = wan_kv_preparation_bias(cfg.dim, revision)
            effective_bias = bias_dtype.bf16? ? bias_values.map { |value| wan_kv_preparation_bf16(value) } : bias_values
            positions_values = wan_kv_preparation_positions(cfg, revision)

            k_norm.write(k_values)
            v_unbiased.write(v_values)
            k_reference.write(k_values)
            v_reference.write(v_values)
            positions.write(positions_values)
            wan_kv_preparation_write_bias(v_bias, bias_values, bias_dtype)
            expanded_bias.write(Array(Float32).new(cfg.values) { |index| effective_bias[index % cfg.dim] })

            chain = ML::Metal::MPSCommandChain.new
            begin
              allocation_start = ML::MetalBuffer.video_spec_allocations
              ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(
                chain.encoder, k_norm, v_unbiased, v_bias, bias_dtype, positions,
                frequency_hi, frequency_lo,
                prepared_k, prepared_v, cfg, trace_k: trace_k, trace_v: trace_v)
              ML::MetalBuffer.video_spec_allocations.should eq(allocation_start),
                "the fused helper must not allocate temporary Metal buffers"

              ML::Video::WanS2VBlock.encode_rope_pair(
                chain.encoder, q_dummy, k_reference, positions, q_rotated, k_rotated,
                cfg, rope,
              )
              ML::Video::MetalOps.add(chain.encoder, v_reference, expanded_bias,
                v_biased, cfg.values)
              ML::Video::MetalAttentionMPS.transpose_kv!(chain, k_rotated, v_biased,
                reference_k, reference_v, cfg.batches, cfg.seq, cfg.heads, cfg.head_dim)
              chain.wait_count.should eq(0), "the fused helper and oracle must not wait"
              chain.commit_and_wait
              chain.wait_count.should eq(1), "one terminal wait must complete both paths"

              wan_kv_preparation_assert_finite(prepared_k, cfg.values, "prepared K")
              wan_kv_preparation_assert_finite(prepared_v, cfg.values, "prepared V")
              wan_kv_preparation_bits(prepared_k, cfg.values).should(eq(
                wan_kv_preparation_bits(reference_k, cfg.values)),
                "packed K must match separate RoPE plus transpose bitwise")
              wan_kv_preparation_bits(prepared_v, cfg.values).should(eq(
                wan_kv_preparation_bits(reference_v, cfg.values)),
                "packed V must match source-dtype bias plus transpose bitwise")

              expected_k_input = trace_k ? wan_kv_preparation_bits(k_rotated, cfg.values) : k_values.map(&.unsafe_as(UInt32))
              expected_v_input = trace_v ? wan_kv_preparation_bits(v_biased, cfg.values) : v_values.map(&.unsafe_as(UInt32))
              wan_kv_preparation_bits(k_norm, cfg.values).should(eq(expected_k_input),
                "trace_k controls only whether the rotated BTHD K is written back")
              wan_kv_preparation_bits(v_unbiased, cfg.values).should(eq(expected_v_input),
                "trace_v controls only whether the biased BTHD V is written back")
            ensure
              chain.commit_and_wait unless chain.completed?
              chain.release if chain.completed?
            end
          end
        ensure
          candidate.release
          reference.release
          rope.release
          inputs.release
        end
      end
    end
  end

  it "rejects unsupported batches, odd head dimensions, aliases, and short buffers before dispatch" do
    ML::Video::WanS2VBlock.prepare!
    cfg = wan_kv_preparation_config(2, 9)
    expect_raises(ArgumentError) do
      ML::Video::WanS2VConfig.new(
        batches: 1, seq: 9, context_tokens: 3, dim: 30, heads: 2,
        ffn_dim: 60, split: 4, theta: 10_000.0_f32,
        video_frames: 1, spatial_tokens: 2,
      )
    end

    inputs = ML::Video::Arena.new
    rope = ML::Video::Arena.new
    malformed = ML::Video::Arena.new
    begin
      ML::Video::WanS2VBlock.prepare_rope_scratch!(rope, cfg)
      k_norm = inputs.upload("k_norm", wan_kv_preparation_values(cfg.values, 1))
      v_unbiased = inputs.upload("v_unbiased", wan_kv_preparation_values(cfg.values, 2))
      v_bias = inputs.upload("v_bias", Array(Float32).new(cfg.dim, 0.0_f32))
      positions = inputs.upload("positions", wan_kv_preparation_positions(cfg, 0))
      prepared_k = inputs.alloc("prepared_k", cfg.values)
      prepared_v = inputs.alloc("prepared_v", cfg.values)
      short = malformed.alloc("short", cfg.values - 1)
      frequency_hi_name = rope.buffers.keys.find { |name| name.includes?("rope_freq_hi") } ||
                          raise("missing prepared RoPE hi frequency buffer")
      frequency_lo_name = rope.buffers.keys.find { |name| name.includes?("rope_freq_lo") } ||
                          raise("missing prepared RoPE lo frequency buffer")
      frequency_hi = rope[frequency_hi_name]
      frequency_lo = rope[frequency_lo_name]
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)

      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(
          encoder, k_norm, v_unbiased, v_bias, ML::Video::WeightDType::F32,
          positions, frequency_hi, frequency_lo, k_norm, prepared_v, cfg,
        )
      end
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(
          encoder, k_norm, v_unbiased, v_bias, ML::Video::WeightDType::F32,
          positions, frequency_hi, frequency_lo, prepared_k, prepared_k, cfg,
        )
      end
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(
          encoder, k_norm, v_unbiased, v_bias, ML::Video::WeightDType::F32,
          positions, frequency_hi, frequency_lo, short, prepared_v, cfg,
        )
      end

      cfg_b2 = ML::Video::WanS2VConfig.new(
        batches: 2, seq: 9, context_tokens: 3, dim: 32, heads: 2,
        ffn_dim: 64, split: 4, theta: 10_000.0_f32,
        video_frames: 1, spatial_tokens: 2,
      )
      b2_rope = ML::Video::Arena.new
      begin
        ML::Video::WanS2VBlock.prepare_rope_scratch!(b2_rope, cfg_b2)
        b2_k = inputs.upload("b2_k", wan_kv_preparation_values(cfg_b2.values, 3))
        b2_v = inputs.upload("b2_v", wan_kv_preparation_values(cfg_b2.values, 4))
        b2_positions = inputs.upload("b2_positions", wan_kv_preparation_positions(cfg_b2, 0))
        b2_out_k = malformed.alloc("b2_out_k", cfg_b2.values)
        b2_out_v = malformed.alloc("b2_out_v", cfg_b2.values)
        b2_hi_name = b2_rope.buffers.keys.find { |name| name.includes?("rope_freq_hi") } ||
                     raise("missing B2 prepared RoPE hi frequencies")
        b2_lo_name = b2_rope.buffers.keys.find { |name| name.includes?("rope_freq_lo") } ||
                     raise("missing B2 prepared RoPE lo frequencies")
        expect_raises(ArgumentError) do
          ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(
            encoder, b2_k, b2_v, v_bias, ML::Video::WeightDType::F32,
            b2_positions, b2_rope[b2_hi_name], b2_rope[b2_lo_name],
            b2_out_k, b2_out_v, cfg_b2,
          )
        end
      ensure
        b2_rope.release
      end

      encoder.end_encoding
      command.discard
    ensure
      malformed.release
      rope.release
      inputs.release
    end
  end
end
