require "./native"
require "digest/sha256"

module ML::Audio::Drama3
  module Wav
    extend self
    MAX_REFERENCE_WAV_BYTES = 1_048_576

    # Host input preparation only: keep the exact signed PCM values so that
    # normalization can be enqueued on the inference device.
    struct Reference
      getter pcm : Vector
      getter sample_rate : Int32
      getter sha256 : String

      def initialize(@pcm : Vector, @sample_rate : Int32, @sha256 : String)
      end

      def samples : Int32
        @pcm.size
      end

      def frames : Int32
        (samples + 2047) // 2048
      end
    end

    def read_reference(path : String) : Reference
      # A short reference bounds encoder activation memory before model work.
      # Read one bounded snapshot so the parsed PCM and reported digest cannot
      # describe different versions of a concurrently replaced file.
      bytes = File.open(path) do |file|
        buffer = Bytes.new(MAX_REFERENCE_WAV_BYTES + 1)
        size = 0
        while size < buffer.size
          count = file.read(buffer[size, buffer.size - size])
          break if count == 0
          size += count
        end
        raise ArgumentError.new("reference WAV exceeds 10 seconds or metadata bound") if size > MAX_REFERENCE_WAV_BYTES
        buffer[0, size].dup
      end
      raise ArgumentError.new("reference must be a RIFF WAVE file") unless bytes.size >= 12 && String.new(bytes[0, 4]) == "RIFF" && String.new(bytes[8, 4]) == "WAVE"
      declared = IO::ByteFormat::LittleEndian.decode(UInt32, bytes[4, 4]).to_i64 + 8
      raise ArgumentError.new("reference RIFF length mismatch") unless declared == bytes.size
      format_seen = false
      data : Bytes? = nil
      offset = 12_i64
      while offset < declared
        raise ArgumentError.new("truncated WAV chunk header") if declared - offset < 8
        start = offset.to_i32
        name = String.new(bytes[start, 4])
        size = IO::ByteFormat::LittleEndian.decode(UInt32, bytes[start + 4, 4]).to_i64
        body = offset + 8
        finish = body + size
        raise ArgumentError.new("truncated WAV chunk") if finish + (size % 2) > declared
        chunk = bytes[body.to_i32, size.to_i32]
        case name
        when "fmt "
          raise ArgumentError.new("duplicate or short WAV format chunk") if format_seen || size < 16
          format = IO::ByteFormat::LittleEndian.decode(UInt16, chunk[0, 2])
          channels = IO::ByteFormat::LittleEndian.decode(UInt16, chunk[2, 2])
          rate = IO::ByteFormat::LittleEndian.decode(UInt32, chunk[4, 4])
          byte_rate = IO::ByteFormat::LittleEndian.decode(UInt32, chunk[8, 4])
          align = IO::ByteFormat::LittleEndian.decode(UInt16, chunk[12, 2])
          bits = IO::ByteFormat::LittleEndian.decode(UInt16, chunk[14, 2])
          unless format == 1 && channels == 1 && rate == 44_100 && bits == 16 && align == 2 && byte_rate == 88_200
            raise ArgumentError.new("reference must be uncompressed mono PCM16 WAV at 44100 Hz")
          end
          format_seen = true
        when "data"
          raise ArgumentError.new("duplicate WAV data chunk") if data
          data = chunk
        end
        offset = finish + (size % 2)
      end
      raise ArgumentError.new("missing WAV format or sample data") unless format_seen && data
      pcm_bytes = data.not_nil!
      raise ArgumentError.new("reference sample data must contain whole PCM16 samples") if pcm_bytes.empty? || pcm_bytes.size.odd?
      count = pcm_bytes.size // 2
      raise ArgumentError.new("reference must be at most 10 seconds") if count > 441_000
      pcm = Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Int16, pcm_bytes[index * 2, 2]).to_f32
      end
      Reference.new(pcm, 44_100, Digest::SHA256.hexdigest(bytes))
    end

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
