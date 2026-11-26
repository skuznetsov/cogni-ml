require "spec"
require "../src/ml/audio/drama3/native"

include ML::Audio::Drama3

describe "DRAMA-3 native math" do
  it "uses row-major output-by-input matrices" do
    Matrix.new(2, 3, [1_f32, 2_f32, 3_f32, -1_f32, 0_f32, 1_f32]).matvec([2_f32, 1_f32, 4_f32]).should eq([16_f32, 2_f32])
  end

  it "keeps zero-centered scales in FP32" do
    x = [0.25_f32, -0.5_f32, 0.75_f32, 1_f32]
    w = [0.001_f32, -0.002_f32, 0.003_f32, -0.004_f32]
    y = Ops.rms_norm(x, w, 1e-6, true)
    y[0].should be_close(0.365513_f32, 2e-6)
    y[3].should be_close(1.45475_f32, 2e-5)
  end

  it "distinguishes interleaved and half-split partial RoPE" do
    a = [1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32]
    b = a.dup
    Ops.rope!(a, 1, 6, 4, 1, 10000.0, true)
    Ops.rope!(b, 1, 6, 4, 1, 10000.0, false)
    a.should_not eq(b)
    a[4..].should eq([5_f32, 6_f32])
    b[4..].should eq([5_f32, 6_f32])
  end

  it "rejects malformed matrices and non-finite activations" do
    expect_raises(ArgumentError) { Matrix.new(2, 2, [1_f32]) }
    expect_raises(ArgumentError) { Matrix.new(1, 2, [1_f32, 2_f32]).matvec([1_f32]) }
    expect_raises(ArgumentError) { Ops.finite!([Float32::NAN], "test") }
  end
end
