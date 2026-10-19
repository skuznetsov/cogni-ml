require "spec"
require "file/tempfile"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_stage_io"
require "./support/video_allocation_counter"

describe ML::Video::StageIO do
  it "reloads overlapping views across GPU retirements with one terminal host wait" do
    count = 257
    stages = 12
    prefix = 257
    bytes = count * sizeof(Float32)
    payload = Bytes.new(prefix + stages * bytes, 0_u8)
    stages.times do |stage|
      values = Array(Float32).new(count) { |i| (stage * 1009 + i).to_f32 }
      payload[prefix + stage * bytes, bytes].copy_from(values.to_unsafe.as(UInt8*), bytes)
    end
    file = File.tempfile("wan-stage-io", ".bin")
    file.write(payload)
    file.close
    pool = ML::Video::MetalStagePool.new(16384_i64)
    views = [] of ML::Video::MetalStageBufferView
    arena = ML::Video::Arena.new
    io = ML::Video::StageIO.new
    chain = ML::Metal::MPSCommandChain.new
    begin
      2.times { |i| views << pool.allocate!("phase#{i}", "weights", bytes.to_i64) }
      output = arena.alloc("output", count * stages)
      pipeline = ML::Metal::ComputePipeline.new("stage_io_capture", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void stage_io_capture(device const float *x [[buffer(0)]],
                                     device float *y [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
          if (i < 257) y[i] = x[i] + 1.0f;
        }
        METAL
      plans = Array.new(stages) do |stage|
        io.plan([ML::Video::StageLoad.new(views[stage % 2].buffer, file.path,
          (prefix + stage * bytes).to_i64, bytes.to_i64)])
      end
      expect_raises(ArgumentError, /source bounds/) do
        io.plan([ML::Video::StageLoad.new(views[0].buffer, file.path,
          payload.size.to_i64, bytes.to_i64)])
      end
      expect_raises(ArgumentError, /destination overlap/) do
        io.plan([ML::Video::StageLoad.new(views[0].buffer, file.path, prefix.to_i64, bytes.to_i64),
                 ML::Video::StageLoad.new(views[1].buffer, file.path, prefix.to_i64, bytes.to_i64)])
      end
      before = ML::MetalBuffer.video_spec_allocations
      plans.each_with_index do |plan, stage|
        chain.pause!
        io.begin!(plan, chain)
        expect_raises(ArgumentError, /already open/) { io.begin!(plan, chain) }
        chain.resume!
        chain.encoder.set_pipeline(pipeline).set_buffer(views[stage % 2].buffer, 0)
          .set_buffer(output, 1, ML::Metal::BufferAccess::Write, offset: (stage * bytes).to_i64).dispatch_1d(count)
        chain.pause!
        io.retire!(chain)
        expect_raises(ArgumentError, /No stage/) { io.retire!(chain) }
        chain.continue_async! unless stage == stages - 1
        chain.resume!
      end
      expect_raises(ArgumentError, /Finish stage/) { io.release }
      expect_raises(ArgumentError, /Complete the compute chain/) { io.finish!(chain) }
      chain.wait_count.should eq(0)
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      chain.commit_and_wait
      io.finish!(chain)
      chain.wait_count.should eq(1)
      chain.segment_count.should eq(stages)
      io.stage_count.should eq(stages)
      output.read(count * stages).should eq(Array(Float32).new(count * stages) do |i|
        ((i // count) * 1009 + i % count).to_f32 + 1.0_f32
      end)
    ensure
      unless chain.completed?
        if io.open?
          # The native encoder may already be paused when an exception occurs.
          begin
            chain.pause!
          rescue ArgumentError
          end
          io.retire!(chain)
        end
        chain.commit_and_wait
      end
      io.finish!(chain) unless io.finished?
      io.release
      chain.release
      arena.release
      views.each { |view| view.release! unless view.released? }
      pool.release
      File.delete(file.path)
    end
  end

  it "rejects a changed source before enqueuing any I/O" do
    file = File.tempfile("wan-stage-io-error", ".bin")
    file.write(Bytes.new(16384, 0_u8))
    file.close
    arena = ML::Video::Arena.new
    io = ML::Video::StageIO.new
    chain = ML::Metal::MPSCommandChain.new
    begin
      buffer = arena.alloc("weights", 4096)
      plan = io.plan([ML::Video::StageLoad.new(buffer, file.path, 0_i64, 16384_i64)])
      File.open(file.path, "w") { }
      chain.pause!
      expect_raises(ArgumentError, /source changed/) { io.begin!(plan, chain) }
      io.open?.should be_false
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      io.finish!(chain)
      io.stage_count.should eq(0)
      io.finished?.should be_true
    ensure
      unless chain.completed?
        if io.open?
          begin
            chain.pause!
          rescue ArgumentError
          end
          io.retire!(chain)
        end
        chain.commit_and_wait
      end
      begin
        io.finish!(chain) unless io.finished?
      rescue ex
        raise ex unless io.finished?
      end
      io.release
      chain.release
      arena.release
      File.delete(file.path)
    end
  end
end
