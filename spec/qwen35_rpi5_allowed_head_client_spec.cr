require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/qwen35_rpi5_allowed_head_client"

private def with_frame_tmp(&)
  dir = File.tempname("qwen35_rpi5_frames_spec")
  Dir.mkdir(dir)
  begin
    yield dir
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

describe ML::GGUF::Qwen35Rpi5AllowedHeadClient do
  it "frames binary stdin requests with little-endian hidden rows" do
    bytes = ML::GGUF::Qwen35Rpi5AllowedHeadClient.frame_bytes(
      [1.0_f32, -2.5_f32],
      [83_i32, 951_i32],
      vocab_rows: 1000,
    )

    bytes[0, 11].should eq("bin\t83,951\n".to_slice)
    bytes[11, 4].should eq(Bytes[0x00, 0x00, 0x80, 0x3f])
    bytes[15, 4].should eq(Bytes[0x00, 0x00, 0x20, 0xc0])
    bytes[19].should eq('\n'.ord)
  end

  it "rejects empty or out-of-range requests before writing a frame" do
    expect_raises(ArgumentError, "hidden must not be empty") do
      ML::GGUF::Qwen35Rpi5AllowedHeadClient.frame_bytes([] of Float32, [1_i32])
    end

    expect_raises(ArgumentError, "allowed_ids must not be empty") do
      ML::GGUF::Qwen35Rpi5AllowedHeadClient.frame_bytes([1.0_f32], [] of Int32)
    end

    expect_raises(ArgumentError, "allowed token id 1000 out of range 0...1000") do
      ML::GGUF::Qwen35Rpi5AllowedHeadClient.frame_bytes([1.0_f32], [1000_i32], vocab_rows: 1000)
    end
  end

  it "parses resident stdin result rows" do
    line = "resident_stdin_result\trequest=1\tallowed=5\tgpu_ms=0.243\tcpu_ms=0.219\tspeedup=0.900x\tmax_abs_diff=2.38419e-06\ttop1_match=true\tgpu_top1_src=13042\tcpu_top1_src=13042\tgpu_top1_logit=1.25\tcpu_top1_logit=1.24999"
    result = ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_result_line?(line).not_nil!

    result.request.should eq(1)
    result.allowed.should eq(5)
    result.gpu_ms.should be_close(0.243, 1.0e-12)
    result.cpu_ms.should be_close(0.219, 1.0e-12)
    result.speedup.should be_close(0.9, 1.0e-12)
    result.max_abs_diff.should be_close(2.38419e-06, 1.0e-12)
    result.top1_match.should be_true
    result.gpu_top1_src.should eq(13042)
    result.cpu_top1_src.should eq(13042)
    result.gpu_top1_logit.should eq(1.25)
    result.cpu_top1_logit.should eq(1.24999)
    result.top1_tuple!([62_i32, 648_i32, 13042_i32, 23256_i32, 47933_i32]).should eq({13042_i32, 1.25_f32})
  end

  it "keeps old resident stdin result rows parseable while the remote probe rolls forward" do
    line = "resident_stdin_result\trequest=1\tallowed=5\tgpu_ms=0.243\tcpu_ms=0.219\tspeedup=0.900x\tmax_abs_diff=2.38419e-06\ttop1_match=true\tgpu_top1_src=13042\tcpu_top1_src=13042"
    result = ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_result_line?(line).not_nil!

    result.gpu_top1_src.should eq(13042)
    result.cpu_top1_src.should eq(13042)
    result.gpu_top1_logit.should be_nil
    result.cpu_top1_logit.should be_nil

    expect_raises(Exception, "resident stdin result missing gpu_top1_logit") do
      result.top1_tuple!([13042_i32])
    end
  end

  it "ignores non-result lines and rejects malformed result rows" do
    output = "replay_rows=2\nresident_stdin_result\trequest=0\tallowed=4\tgpu_ms=0.160\tcpu_ms=0.154\tspeedup=0.958x\tmax_abs_diff=2.86102e-06\ttop1_match=true\tgpu_top1_src=13766\tcpu_top1_src=13766\nthrottled=0x0\n"
    ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_results(output).size.should eq(1)

    expect_raises(ArgumentError, "resident stdin result missing cpu_top1_src") do
      ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_result_line?(
        "resident_stdin_result\trequest=0\tallowed=4\tgpu_ms=0.160\tcpu_ms=0.154\tspeedup=0.958x\tmax_abs_diff=0\ttop1_match=true\tgpu_top1_src=13766"
      )
    end
  end

  it "fails closed when converting invalid resident rows to the forward_top1_allowed tuple" do
    mismatch = ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_result_line?(
      "resident_stdin_result\trequest=1\tallowed=2\tgpu_ms=0.1\tcpu_ms=0.1\tspeedup=1.0x\tmax_abs_diff=0\ttop1_match=false\tgpu_top1_src=1\tcpu_top1_src=2\tgpu_top1_logit=3.0\tcpu_top1_logit=4.0"
    ).not_nil!

    expect_raises(Exception, "resident stdin top1 mismatch") do
      mismatch.top1_tuple!([1_i32, 2_i32])
    end

    outside = ML::GGUF::Qwen35Rpi5AllowedHeadClient.parse_result_line?(
      "resident_stdin_result\trequest=1\tallowed=2\tgpu_ms=0.1\tcpu_ms=0.1\tspeedup=1.0x\tmax_abs_diff=0\ttop1_match=true\tgpu_top1_src=3\tcpu_top1_src=3\tgpu_top1_logit=3.0\tcpu_top1_logit=3.0"
    ).not_nil!

    expect_raises(Exception, "resident stdin top1 3 outside allowed set") do
      outside.top1_tuple!([1_i32, 2_i32])
    end
  end

  it "exports capture replay batches as resident stdin frames" do
    with_frame_tmp do |dir|
      f32_path = File.join(dir, "x.f32")
      File.open(f32_path, "wb") do |io|
        [1.0_f32, -2.5_f32, 0.25_f32, 4.0_f32].each do |value|
          io.write_bytes(value, IO::ByteFormat::LittleEndian)
        end
      end

      output = IO::Memory.new
      error = IO::Memory.new
      status = Process.run(
        "crystal",
        ["scripts/rpi5_q6_resident_stdin_frames.cr", f32_path, "2", "83,951:62", "2"],
        output: output,
        error: error,
      )

      status.success?.should be_true, error.to_s
      bytes = output.to_slice
      bytes[0, 11].should eq("bin\t83,951\n".to_slice)
      bytes[11, 4].should eq(Bytes[0x00, 0x00, 0x80, 0x3f])
      bytes[15, 4].should eq(Bytes[0x00, 0x00, 0x20, 0xc0])
      bytes[19].should eq('\n'.ord)
      bytes[20, 7].should eq("bin\t62\n".to_slice)
      bytes[27, 4].should eq(Bytes[0x00, 0x00, 0x80, 0x3e])
      bytes[31, 4].should eq(Bytes[0x00, 0x00, 0x80, 0x40])
      bytes[35].should eq('\n'.ord)
    end
  end

  it "fails closed when frame export metadata does not match the batch" do
    with_frame_tmp do |dir|
      f32_path = File.join(dir, "x.f32")
      File.open(f32_path, "wb") do |io|
        io.write_bytes(1.0_f32, IO::ByteFormat::LittleEndian)
      end

      error = IO::Memory.new
      status = Process.run(
        "crystal",
        ["scripts/rpi5_q6_resident_stdin_frames.cr", f32_path, "2", "83", "1"],
        output: Process::Redirect::Close,
        error: error,
      )

      status.success?.should be_false
      error.to_s.should contain("hidden batch byte size 4 does not match expected 8")
    end
  end
end
