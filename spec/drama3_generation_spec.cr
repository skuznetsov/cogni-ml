require "spec"
require "../src/ml/audio/drama3/generator"
require "../src/ml/audio/drama3/wav"

include ML::Audio::Drama3

describe "DRAMA-3 generation boundaries" do
  it "uses full-softmax probabilities and drops the top-p crossing token" do
    sampler = Sampler.new(42_u64)
    # Full probabilities are .5, .3, .2. At .7 the second token crosses the
    # threshold, so only the first survives. A shifted top-p mask keeps two.
    logits = [Math.log(0.5).to_f32, Math.log(0.3).to_f32, Math.log(0.2).to_f32]
    sampler.candidates(logits, 1.0, 2, 0.7).map(&.[0]).should eq([0])
    sampler.candidates(logits, 1.0, 2, 0.85).map(&.[0]).should eq([0, 1])
    sampler.candidates(logits, 1.0, 2, 0.01).map(&.[0]).should eq([0])
    expect_raises(ArgumentError) { sampler.sample(logits, 1.0, 0, 0.9) }
    expect_raises(ArgumentError) { sampler.sample([Float32::NAN], 1.0, 1, 0.9) }
  end

  it "writes finite mono PCM16 with a consistent RIFF length" do
    path = File.tempname("drama3", ".wav")
    begin
      Wav.write(path, [-1_f32, 0_f32, 0.5_f32, 1_f32])
      bytes = File.read(path).to_slice
      String.new(bytes[0, 4]).should eq("RIFF")
      IO::ByteFormat::LittleEndian.decode(UInt32, bytes[4, 4]).should eq(44_u32)
      IO::ByteFormat::LittleEndian.decode(UInt32, bytes[24, 4]).should eq(44100_u32)
      IO::ByteFormat::LittleEndian.decode(Int16, bytes[44, 2]).should eq(-32767_i16)
      bytes.size.should eq(52)
      expect_raises(ArgumentError) { Wav.write(path, [Float32::INFINITY]) }
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
