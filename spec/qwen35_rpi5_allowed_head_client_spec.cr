require "./spec_helper"
require "../src/ml/gguf/qwen35_rpi5_allowed_head_client"

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
    line = "resident_stdin_result\trequest=1\tallowed=5\tgpu_ms=0.243\tcpu_ms=0.219\tspeedup=0.900x\tmax_abs_diff=2.38419e-06\ttop1_match=true\tgpu_top1_src=13042\tcpu_top1_src=13042"
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
end
