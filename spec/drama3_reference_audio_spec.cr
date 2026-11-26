require "spec"
require "../src/ml/audio/drama3/wav"
require "../src/ml/audio/drama3/reference_audio"

private def drama3_reference_wav_bytes(rate = 44100_u32, channels = 1_u16, bits = 16_u16,
                                       format = 1_u16, samples = [-32768_i16, 0_i16, 32767_i16]) : Bytes
  body = IO::Memory.new
  body << "WAVEJUNK"
  body.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
  body << "abc\0fmt "
  body.write_bytes(16_u32, IO::ByteFormat::LittleEndian)
  body.write_bytes(format, IO::ByteFormat::LittleEndian)
  body.write_bytes(channels, IO::ByteFormat::LittleEndian)
  body.write_bytes(rate, IO::ByteFormat::LittleEndian)
  align = channels * (bits // 8)
  body.write_bytes(rate * align, IO::ByteFormat::LittleEndian)
  body.write_bytes(align, IO::ByteFormat::LittleEndian)
  body.write_bytes(bits, IO::ByteFormat::LittleEndian)
  body << "data"
  body.write_bytes((samples.size * 2).to_u32, IO::ByteFormat::LittleEndian)
  samples.each { |sample| body.write_bytes(sample, IO::ByteFormat::LittleEndian) }
  io = IO::Memory.new
  io << "RIFF"
  io.write_bytes(body.size.to_u32, IO::ByteFormat::LittleEndian)
  io.write(body.to_slice)
  io.to_slice.dup
end

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe ML::Audio::Drama3::ReferenceAudio do
    it "normalizes signed PCM16 endpoints on GPU without a handoff read" do
      with_drama3_reference_wav(drama3_reference_wav_bytes) do |filename|
        context = ML::Audio::Drama3::Resident::Context.new
        output = ML::Audio::Drama3::ReferenceAudio.upload(ML::Audio::Drama3::Wav.read_reference(filename), context)
        context.readbacks.should be_empty
        context.graph_stats["submitted"].should eq(0_i64)
        context.read(output, "terminal").should eq([-1_f32, 0_f32, 32767_f32 / 32768_f32])
        context.readbacks["diagnostic"].should eq(0_i64)
      end
    end
  end
{% end %}

private def with_drama3_reference_wav(bytes : Bytes, &)
  filename = File.tempname("drama3-reference", ".wav")
  begin
    File.write(filename, bytes)
    yield filename
  ensure
    File.delete(filename) if File.exists?(filename)
  end
end

describe "DRAMA-3 reference audio input" do
  it "parses PCM values without host normalization and skips padded ancillary chunks" do
    bytes = drama3_reference_wav_bytes
    with_drama3_reference_wav(bytes) do |filename|
      input = ML::Audio::Drama3::Wav.read_reference(filename)
      input.sample_rate.should eq(44100)
      input.pcm.should eq([-32768_f32, 0_f32, 32767_f32])
      input.samples.should eq(3)
      input.frames.should eq(1)
      input.sha256.should eq(Digest::SHA256.hexdigest(bytes))
    end
  end

  it "rejects rate, channel, encoding and empty inputs before GPU work" do
    [drama3_reference_wav_bytes(rate: 24000_u32),
     drama3_reference_wav_bytes(channels: 2_u16),
     drama3_reference_wav_bytes(bits: 32_u16),
     drama3_reference_wav_bytes(format: 3_u16),
     drama3_reference_wav_bytes(samples: [] of Int16)].each do |bytes|
      with_drama3_reference_wav(bytes) do |filename|
        expect_raises(ArgumentError) { ML::Audio::Drama3::Wav.read_reference(filename) }
      end
    end
  end

  it "rejects inconsistent RIFF lengths and truncated sample data" do
    [drama3_reference_wav_bytes[0...-1], drama3_reference_wav_bytes + Bytes[0], Bytes[0, 1, 2]].each do |bytes|
      with_drama3_reference_wav(bytes) do |filename|
        expect_raises(ArgumentError) { ML::Audio::Drama3::Wav.read_reference(filename) }
      end
    end
  end

  it "bounds reference length before allocating model activations" do
    bytes = drama3_reference_wav_bytes(samples: Array(Int16).new(441001, 0_i16))
    with_drama3_reference_wav(bytes) do |filename|
      expect_raises(ArgumentError, /10 seconds/) { ML::Audio::Drama3::Wav.read_reference(filename) }
    end
  end

  it "rejects references above the metadata byte bound with a bounded read" do
    bytes = Bytes.new(ML::Audio::Drama3::Wav::MAX_REFERENCE_WAV_BYTES + 1, 0_u8)
    with_drama3_reference_wav(bytes) do |filename|
      expect_raises(ArgumentError, /metadata bound/) { ML::Audio::Drama3::Wav.read_reference(filename) }
    end
  end
end
