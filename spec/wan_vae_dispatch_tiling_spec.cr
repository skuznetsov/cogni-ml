require "spec"
require "../src/ml/video/wan_vae"
require "./support/video_allocation_counter"

private def wan_vae_dispatch_tiling_pipeline(name : String) : ML::Metal::ComputePipeline
  ML::Metal::PipelineCache.get(name) do
    ML::Metal::ComputePipeline.new(name, ML::Video::WanVAE::SOURCE)
  end
end

private def wan_vae_dispatch_tiling_values(count : Int32, multiplier : Int32,
                                           modulus : Int32, shift : Int32,
                                           scale : Float32) : Array(Float32)
  Array(Float32).new(count) do |index|
    (((index * multiplier + shift) % modulus) - modulus // 2).to_f32 * scale
  end
end

private def wan_vae_dispatch_tiling_bf16(arena : ML::Video::Arena, name : String,
                                         values : Array(Float32)) : ML::MetalBuffer
  buffer = arena.alloc_weight(name, values.size, ML::Video::WeightDType::BF16)
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    word = (value.unsafe_as(UInt32) >> 16).to_u16
    bytes[index * 2] = (word & 0xff).to_u8
    bytes[index * 2 + 1] = (word >> 8).to_u8
  end
  buffer.write_bytes(bytes.to_unsafe, bytes.size)
  buffer
end

private def wan_vae_dispatch_tiling_assert_same(name : String,
                                                full : Array(Float32),
                                                tiled : Array(Float32)) : Nil
  tiled.size.should eq(full.size), name
  full.each_with_index do |value, index|
    value.finite?.should be_true, "#{name}: full output is non-finite at #{index}"
    tiled[index].finite?.should be_true, "#{name}: tiled output is non-finite at #{index}"
    tiled[index].unsafe_as(UInt32).should eq(value.unsafe_as(UInt32)),
      "#{name}: bit mismatch at #{index}: full=#{value}, tiled=#{tiled[index]}"
  end
end

private def wan_vae_dispatch_tiling_bind_attention(e : ML::Metal::ComputeEncoder,
                                                   pipeline : ML::Metal::ComputePipeline,
                                                   qkv : ML::MetalBuffer,
                                                   output : ML::MetalBuffer,
                                                   batches : Int32, channels : Int32,
                                                   frames : Int32, height : Int32,
                                                   width : Int32, offset : Int32) : Nil
  e.set_pipeline(pipeline)
  e.set_buffer(qkv, 0)
  e.set_buffer(output, 1, ML::Metal::BufferAccess::Write)
  e.set_value(batches.to_u32, 2)
  e.set_value(channels.to_u32, 3)
  e.set_value(frames.to_u32, 4)
  e.set_value(height.to_u32, 5)
  e.set_value(width.to_u32, 6)
  e.set_value(offset.to_u32, 7)
end

private def wan_vae_dispatch_tiling_bind_conv(e : ML::Metal::ComputeEncoder,
                                              pipeline : ML::Metal::ComputePipeline,
                                              input : ML::MetalBuffer,
                                              cache : ML::MetalBuffer,
                                              weight : ML::MetalBuffer,
                                              bias : ML::MetalBuffer,
                                              output : ML::MetalBuffer,
                                              batches : Int32, in_channels : Int32,
                                              out_channels : Int32, input_t : Int32,
                                              input_h : Int32, input_w : Int32,
                                              cache_t : Int32, kernel_t : Int32,
                                              kernel_h : Int32, kernel_w : Int32,
                                              output_t : Int32, output_h : Int32,
                                              output_w : Int32, bf16 : Bool,
                                              output_offset : Int32) : Nil
  e.set_pipeline(pipeline)
  e.set_buffer(input, 0)
  e.set_buffer(cache, 1)
  # The unused-typed pointer slot aliases the same production weight buffer;
  # the kernel reads only the slot selected by the explicit BF16 flag.
  e.set_buffer(weight, 2)
  e.set_buffer(weight, 3)
  e.set_buffer(bias, 4)
  e.set_buffer(bias, 5)
  e.set_buffer(output, 6, ML::Metal::BufferAccess::Write)
  e.set_value(batches.to_u32, 7)
  e.set_value(in_channels.to_u32, 8)
  e.set_value(out_channels.to_u32, 9)
  e.set_value(input_t.to_u32, 10)
  e.set_value(input_h.to_u32, 11)
  e.set_value(input_w.to_u32, 12)
  e.set_value(cache_t.to_u32, 13)
  e.set_value(kernel_t.to_u32, 14)
  e.set_value(kernel_h.to_u32, 15)
  e.set_value(kernel_w.to_u32, 16)
  e.set_value(1_u32, 17)
  e.set_value(1_u32, 18)
  e.set_value(1_u32, 19)
  e.set_value(0_u32, 20)
  e.set_value(1_u32, 21)
  e.set_value(1_u32, 22)
  e.set_value(output_t.to_u32, 23)
  e.set_value(output_h.to_u32, 24)
  e.set_value(output_w.to_u32, 25)
  e.set_value(bf16 ? 1_u32 : 0_u32, 26)
  e.set_value(bf16 ? 1_u32 : 0_u32, 27)
  e.set_value(output_offset.to_u32, 28)
end

describe "Wan VAE independent-output dispatch tiling" do
  it "matches full and 32-query attention dispatches across frame and batch tails" do
    arena = ML::Video::Arena.new
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      batches = 2
      channels = 32
      frames = 2
      height = 5
      width = 7
      spatial = height * width
      query_count = batches * frames * spatial
      qkv_count = batches * 3 * channels * frames * spatial
      output_count = batches * channels * frames * spatial
      qkv = arena.upload("vae.tiling.attention.qkv",
        wan_vae_dispatch_tiling_values(qkv_count, 29, 113, 7, 0.007_f32))
      full = arena.upload("vae.tiling.attention.full", Array.new(output_count, Float32::NAN))
      tiled = arena.upload("vae.tiling.attention.tiled", Array.new(output_count, Float32::NAN))
      pipeline = wan_vae_dispatch_tiling_pipeline("wan_vae_attention_f32")

      active = ML::Metal::CommandBuffer.new
      command = active
      e = ML::Metal::ComputeEncoder.new(active)
      before = ML::MetalBuffer.video_spec_allocations

      wan_vae_dispatch_tiling_bind_attention(e, pipeline, qkv, full,
        batches, channels, frames, height, width, 0)
      e.dispatch_threadgroups({query_count, 1, 1}, {channels, 1, 1})

      chunk_rows = 32
      offset = 0
      while offset < query_count
        rows = Math.min(chunk_rows, query_count - offset)
        wan_vae_dispatch_tiling_bind_attention(e, pipeline, qkv, tiled,
          batches, channels, frames, height, width, offset)
        e.dispatch_threadgroups({rows, 1, 1}, {channels, 1, 1})
        offset += rows
      end

      ML::MetalBuffer.video_spec_allocations.should eq(before),
        "attention dispatch encoding must allocate no MetalBuffers"
      e.end_encoding
      active.commit
      active.wait # Single terminal wait for both full and tiled executions.

      wan_vae_dispatch_tiling_assert_same("attention", full.read(output_count), tiled.read(output_count))
    ensure
      if active = command
        if active.committed?
          active.wait unless active.completed?
        else
          active.discard
        end
      end
      arena.release
    end
  end

  it "matches full and output-offset Conv3d dispatches with cache, batch, frame and BF16 tails" do
    arena = ML::Video::Arena.new
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      batches = 2
      in_channels = 2
      out_channels = 3
      input_t = 3
      input_h = 3
      input_w = 5
      cache_t = 2
      kernel_t = 3
      kernel_h = 3
      kernel_w = 3
      output_t = 3
      output_h = 3
      output_w = 5
      output_count = batches * out_channels * output_t * output_h * output_w

      input = arena.upload("vae.tiling.conv.input",
        wan_vae_dispatch_tiling_values(batches * in_channels * input_t * input_h * input_w,
          17, 97, 19, 0.013_f32))
      cache = arena.upload("vae.tiling.conv.cache",
        wan_vae_dispatch_tiling_values(batches * in_channels * cache_t * input_h * input_w,
          31, 89, 23, 0.011_f32))
      weight_values = wan_vae_dispatch_tiling_values(
        out_channels * in_channels * kernel_t * kernel_h * kernel_w,
        13, 71, 5, 0.003_f32)
      bias_values = wan_vae_dispatch_tiling_values(out_channels, 7, 19, 3, 0.01_f32)

      weight_f32 = arena.upload("vae.tiling.conv.weight.f32", weight_values)
      bias_f32 = arena.upload("vae.tiling.conv.bias.f32", bias_values)
      weight_bf16 = wan_vae_dispatch_tiling_bf16(arena, "vae.tiling.conv.weight.bf16", weight_values)
      bias_bf16 = wan_vae_dispatch_tiling_bf16(arena, "vae.tiling.conv.bias.bf16", bias_values)

      full_f32 = arena.upload("vae.tiling.conv.full.f32", Array.new(output_count, Float32::NAN))
      tiled_f32 = arena.upload("vae.tiling.conv.tiled.f32", Array.new(output_count, Float32::NAN))
      full_bf16 = arena.upload("vae.tiling.conv.full.bf16", Array.new(output_count, Float32::NAN))
      tiled_bf16 = arena.upload("vae.tiling.conv.tiled.bf16", Array.new(output_count, Float32::NAN))
      pipeline = wan_vae_dispatch_tiling_pipeline("wan_vae_conv3d_f32")

      active = ML::Metal::CommandBuffer.new
      command = active
      e = ML::Metal::ComputeEncoder.new(active)
      before = ML::MetalBuffer.video_spec_allocations

      [{weight_f32, bias_f32, full_f32, false},
       {weight_bf16, bias_bf16, full_bf16, true}].each do |entry|
        weight = entry[0]
        bias = entry[1]
        output = entry[2]
        bf16 = entry[3]
        wan_vae_dispatch_tiling_bind_conv(e, pipeline, input, cache, weight, bias, output,
          batches, in_channels, out_channels, input_t, input_h, input_w, cache_t,
          kernel_t, kernel_h, kernel_w, output_t, output_h, output_w, bf16, 0)
        e.dispatch_1d(output_count, 64)
      end

      chunk_size = 64
      [{weight_f32, bias_f32, tiled_f32, false},
       {weight_bf16, bias_bf16, tiled_bf16, true}].each do |entry|
        weight = entry[0]
        bias = entry[1]
        output = entry[2]
        bf16 = entry[3]
        offset = 0
        while offset < output_count
          count = Math.min(chunk_size, output_count - offset)
          wan_vae_dispatch_tiling_bind_conv(e, pipeline, input, cache, weight, bias, output,
            batches, in_channels, out_channels, input_t, input_h, input_w, cache_t,
            kernel_t, kernel_h, kernel_w, output_t, output_h, output_w, bf16, offset)
          e.dispatch_1d(count, 64)
          offset += count
        end
      end

      ML::MetalBuffer.video_spec_allocations.should eq(before),
        "Conv3d dispatch encoding must allocate no MetalBuffers"
      e.end_encoding
      active.commit
      active.wait # Single terminal wait for F32 and BF16 full/tiled executions.

      wan_vae_dispatch_tiling_assert_same("Conv3d F32",
        full_f32.read(output_count), tiled_f32.read(output_count))
      wan_vae_dispatch_tiling_assert_same("Conv3d BF16",
        full_bf16.read(output_count), tiled_bf16.read(output_count))
    ensure
      if active = command
        if active.committed?
          active.wait unless active.completed?
        else
          active.discard
        end
      end
      arena.release
    end
  end
end
