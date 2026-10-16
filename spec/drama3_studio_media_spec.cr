require "spec"
require "file_utils"
require "../src/ml/audio/drama3/studio/media"

private def with_studio_media_dir(& : String ->)
  dir = File.tempname("drama3-studio-media")
  Dir.mkdir(dir)
  yield dir
ensure
  FileUtils.rm_rf(dir) if dir
end

describe ML::Audio::Drama3::Studio::Media do
  it "joins takes in order without inserted PCM silence and exports real compressed streams" do
    with_studio_media_dir do |dir|
      first = File.join(dir, "first.wav")
      second = File.join(dir, "second.wav")
      ML::Audio::Drama3::Wav.write(first, Array(Float32).new(4410, 0.25_f32))
      ML::Audio::Drama3::Wav.write(second, Array(Float32).new(4410, -0.125_f32))
      media = ML::Audio::Drama3::Studio::Media.new
      output = File.join(dir, "joined.wav")
      media.export([first, second], output, "wav").should be_close(0.2, 1e-6)
      decoded = IO::Memory.new
      Process.run(media.ffmpeg, ["-v", "error", "-i", output, "-f", "s16le", "-"], output: decoded).success?.should be_true
      decoded.to_slice.size.should eq(17_640)
      IO::ByteFormat::LittleEndian.decode(Int16, decoded.to_slice[0, 2]).should eq(8192)
      IO::ByteFormat::LittleEndian.decode(Int16, decoded.to_slice[8820, 2]).should eq(-4096)
      {"mp3" => "mp3", "opus" => "opus"}.each do |format, codec|
        path = File.join(dir, "joined.#{format}")
        media.export([first, second], path, format).should be_close(0.2, 1e-6)
        check = IO::Memory.new
        Process.run("ffprobe", ["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "json", path], output: check).success?.should be_true
        JSON.parse(check.to_s)["streams"][0]["codec_name"].as_s.should eq(codec)
      end
    end
  end

  it "converts short references but rejects long references and failed exports" do
    with_studio_media_dir do |dir|
      media = ML::Audio::Drama3::Studio::Media.new
      input = File.join(dir, "input.wav")
      normalized = File.join(dir, "reference.wav")
      ML::Audio::Drama3::Wav.write(input, Array(Float32).new(4410, 0.1_f32))
      media.prepare_reference(input, normalized).should be_close(0.1, 1e-6)
      ML::Audio::Drama3::Wav.read_reference(normalized).samples.should eq(4410)
      long = File.join(dir, "long.wav")
      ML::Audio::Drama3::Wav.write(long, Array(Float32).new(450_000, 0.1_f32))
      expect_raises(ArgumentError) { media.prepare_reference(long, File.join(dir, "rejected.wav")) }
      File.exists?(File.join(dir, "rejected.wav")).should be_false
      expect_raises(ArgumentError) { media.export([File.join(dir, "missing.wav")], File.join(dir, "bad.wav"), "wav") }
      expect_raises(ArgumentError) { media.export([input], File.join(dir, "bad.aac"), "aac") }
      expect_raises(ArgumentError) { media.export([input], input, "wav") }
      File.exists?(input).should be_true
      expect_raises(ArgumentError) { media.prepare_reference(input, normalized) }
      File.exists?(normalized).should be_true
      failed = ML::Audio::Drama3::Studio::Media.new("/usr/bin/false")
      expect_raises(ArgumentError) { failed.export([input], File.join(dir, "failed.wav"), "wav") }
      File.exists?(File.join(dir, "failed.wav")).should be_false
    end
  end

  it "bounds conversion wall time and stderr while cleaning partial output and preserving existing output" do
    with_studio_media_dir do |dir|
      expect_raises(ArgumentError) do
        ML::Audio::Drama3::Studio::Media.new("/usr/bin/false", Time::Span.zero)
      end

      fake_ffmpeg = File.join(dir, "ffmpeg-flood")
      File.write(fake_ffmpeg, <<-SH)
        #!/bin/sh
        for output_path do :; done
        printf '%s\\n' "$$" > "${output_path}.pid"
        printf 'diagnostic-prefix\\n' >&2
        printf 'partial output' > "$output_path"
        index=0
        while [ "$index" -lt 256 ]; do
          printf '%4096s\\n' '' >&2
          index=$((index + 1))
        done
        trap '' TERM
        while :; do :; done
        SH
      File.chmod(fake_ffmpeg, 0o700)

      input = File.join(dir, "input.wav")
      ML::Audio::Drama3::Wav.write(input, Array(Float32).new(4410, 0.1_f32))
      media = ML::Audio::Drama3::Studio::Media.new(fake_ffmpeg, 250.milliseconds)
      partial = File.join(dir, "partial.wav")
      started = Time.instant
      error = expect_raises(ArgumentError) { media.prepare_reference(input, partial) }
      elapsed = Time.instant - started

      error.message.not_nil!.should contain("timed out")
      error.message.not_nil!.should contain("diagnostic-prefix")
      error.message.not_nil!.bytesize.should be <= 4 * 1024 + 128
      elapsed.should be < 2.seconds
      File.exists?(partial).should be_false
      Process.exists?(File.read(File.join(dir, "partial.wav.pid")).strip.to_i64).should be_false

      existing = File.join(dir, "existing.wav")
      File.write(existing, "preserve-existing-output")
      expect_raises(ArgumentError) { media.prepare_reference(input, existing) }
      File.read(existing).should eq("preserve-existing-output")
    end
  end
end
