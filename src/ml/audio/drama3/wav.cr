require "./native"

module ML::Audio::Drama3
  module Wav
    extend self

    def write(path : String, samples : Vector, sample_rate : Int32 = 44_100) : Nil
      raise ArgumentError.new("empty waveform") if samples.empty?
      raise ArgumentError.new("invalid sample rate") unless sample_rate > 0 && sample_rate <= 384000
      Ops.finite!(samples, "WAV input")
      raise ArgumentError.new("waveform too large for RIFF PCM16") if samples.size.to_i64 * 2 > UInt32::MAX - 36
      File.open(path, "wb") do |io|
        io << "RIFF"
        io.write_bytes((36_u32 + samples.size.to_u32 * 2), IO::ByteFormat::LittleEndian)
        io << "WAVEfmt "
        io.write_bytes(16_u32, IO::ByteFormat::LittleEndian)
        io.write_bytes(1_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(1_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(sample_rate.to_u32, IO::ByteFormat::LittleEndian)
        io.write_bytes(sample_rate.to_u32 * 2, IO::ByteFormat::LittleEndian)
        io.write_bytes(2_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(16_u16, IO::ByteFormat::LittleEndian)
        io << "data"
        io.write_bytes(samples.size.to_u32 * 2, IO::ByteFormat::LittleEndian)
        samples.each do |value|
          pcm = (value.clamp(-1_f32, 1_f32) * 32767_f32).round.to_i16
          io.write_bytes(pcm, IO::ByteFormat::LittleEndian)
        end
      end
    end
  end
end
