# Paired qualification of the native and coalesced DAC convolution layouts.
#
# Build: crystal build examples/drama3_dac_layout_bench.cr -Ddrama3_metal \
#          -o /tmp/drama3_dac_layout_bench
# Run:   DRAMA3_BACKEND=metal /tmp/drama3_dac_layout_bench MODEL_DIR [--timings=8]
#
# Timed decodes have no stage observers or diagnostic commands. The later
# traced decodes compare exact stage bits on Metal and read only compact counts.
require "digest/sha256"
require "json"
require "../src/ml/audio/drama3/codec_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3DacLayoutBenchmark
    include ML::Audio::Drama3

    alias Context = Resident::Context
    alias Buffer = Resident::Buffer

    SOURCE = {{ read_file("#{__DIR__}/support/drama3_dac_layout_compare.metal") }}
    FIXTURE_PATH = "/opt/mldata_ssd/cogni-ml/drama3-runs/gpu-kernels-20261016/timeline-native-arm0-repeat1.wav.json"
    FIXTURE_SHA256 = "40e627444dc0f9acf254c6fdc5b96bb54f6d6189f52f27d71cf83ed2baf4d44f"
    CODEC_SHA256 = "543445a2ebc5224d2af5956d422f88c2cd21f4018c1a26d81b51ec231bb630f2"
    TIMING_ORDER = ["native", "coalesced", "coalesced", "native", "coalesced", "native", "native", "coalesced"]
    EXPECTED_COMMANDS_PER_DECODE = 32_i64
    STAGE_COUNT = 58

    record Stage, name : String, rows : Int32, channels : Int32, values : Buffer
    record Timing, order_index : Int32, layout : String, seconds : Float64,
      submitted : Int64, completed : Int64, waits : Int64

    def self.expected_names : Array(String)
      names = [] of String
      10.times do |book|
        names << "quantizer.book.#{book}.embedding" << "quantizer.book.#{book}.projection"
      end
      names << "quantizer.codebook_sum"
      8.times { |layer| names << "quantizer.post_module.layers.#{layer}" }
      names << "quantizer.post_module"
      2.times do |stage|
        names << "quantizer.upsample.#{stage}.deconv" << "quantizer.upsample.#{stage}.convnext"
      end
      names << "decoder.model.0"
      4.times { |stage| 5.times { |block| names << "decoder.model.#{stage + 1}.block.#{block}" } }
      names << "decoder.model.5" << "decoder.model.6" << "waveform"
      raise "internal DAC stage list length" unless names.size == STAGE_COUNT
      names
    end

    def self.sha_file(path : String) : String
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        chunk = Bytes.new(1024 * 1024)
        while (size = io.read(chunk)) > 0
          digest.update(chunk[0, size])
        end
      end
      digest.hexfinal
    end

    def self.f32_sha(values : Array(Float32)) : String
      io = IO::Memory.new
      values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def self.load_fixture : NamedTuple(document: JSON::Any, frames: Int32, codes: Array(Float32), codes_sha256: String)
      raise "pinned DAC fixture missing: #{FIXTURE_PATH}" unless File.file?(FIXTURE_PATH)
      raw = File.read(FIXTURE_PATH)
      fixture_sha = Digest::SHA256.hexdigest(raw)
      raise "pinned DAC fixture digest mismatch: #{fixture_sha}" unless fixture_sha == FIXTURE_SHA256
      document = JSON.parse(raw)
      rows = document["frames"].as_a
      raise "expected the pinned 60-frame DAC input" unless rows.size == 60
      codes = [] of Float32
      rows.each_with_index do |row, frame|
        values = row.as_a
        raise "frame #{frame} must contain main plus ten codebooks" unless values.size == 11
        values.each_with_index do |value, column|
          code = value.as_i
          if column == 1
            raise "semantic code out of range at frame #{frame}" unless 0 <= code < 4096
          elsif column >= 2
            raise "residual code out of range at frame #{frame}, book #{column - 1}" unless 0 <= code < 1024
          end
          codes << code.to_f32
        end
      end
      raise "frame-major code shape" unless codes.size == 60 * 11
      {document: document, frames: rows.size.to_i32, codes: codes, codes_sha256: f32_sha(codes)}
    end

    def self.configure_environment! : Nil
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV["DRAMA3_DAC_SCRATCH"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_HOST_PROFILE"] = "0"
      ENV["DRAMA3_MEMORY_PROFILE"] = "0"
      ENV["DRAMA3_ALLOCATION_PROFILE"] = "0"
      ENV.delete("DRAMA3_PROFILE")
      Profile.configure_from_env!
      raise "profiling must remain disabled" if Profile.enabled?
    end

    def self.new_decoder(store : WeightStore, context : Context, layout : String) : CodecResident
      ENV["DRAMA3_DAC_CONV_LAYOUT"] = layout
      decoder = CodecResident.new(store, context)
      raise "decoder did not freeze requested layout" unless decoder.dac_conv_layout == layout
      decoder
    end

    def self.measure_decode(context : Context, decoder : CodecResident, codes : Buffer,
                            frames : Int32, index : Int32, layout : String) : Timing
      before = context.graph_stats
      reads_before = context.readbacks.dup
      raise "timing decode began with pending work" unless before["pending"] == 0
      started = Time.instant
      decoder.decode(codes, frames, false)
      seconds = (Time.instant - started).total_seconds
      after = context.graph_stats
      submitted = after["submitted"] - before["submitted"]
      completed = after["completed"] - before["completed"]
      waits = after["completion_waits"] - before["completion_waits"]
      unless submitted == EXPECTED_COMMANDS_PER_DECODE && completed == submitted && waits == submitted && after["pending"] == 0
        raise "#{layout} decode graph delta invalid: submit=#{submitted} complete=#{completed} waits=#{waits} pending=#{after["pending"]}"
      end
      raise "timed decode added readbacks or diagnostics" unless context.readbacks == reads_before
      Timing.new(index, layout, seconds, submitted, completed, waits)
    end

    def self.capture_native(context : Context, decoder : CodecResident, codes : Buffer, frames : Int32,
                            names : Array(String)) : Array(Stage)
      raise "native capture began with readbacks" unless context.readbacks.empty?
      stages = [] of Stage
      decoder.device_trace = ->(name : String, values : Buffer, rows : Int32, channels : Int32) {
        index = stages.size
        raise "unexpected/duplicate native stage #{name}" unless names[index]? == name
        raise "invalid native stage dimensions #{name}" unless rows > 0 && channels > 0 && values.count == rows * channels
        stages << Stage.new(name, rows, channels, values)
        nil
      }
      decoder.decode(codes, frames, false)
      decoder.device_trace = nil
      raise "native capture read back activations" unless context.readbacks.empty?
      raise "incomplete native stage capture" unless stages.map(&.name) == names
      stages
    end

    def self.compare_coalesced(context : Context, decoder : CodecResident, codes : Buffer,
                               frames : Int32, native_stages : Array(Stage), names : Array(String))
      receipt = context.upload(Array(Float32).new(STAGE_COUNT * 2, 0_f32))
      observed = [] of String
      decoder.device_trace = ->(name : String, actual : Buffer, rows : Int32, channels : Int32) {
        index = observed.size
        raise "unexpected/duplicate coalesced stage #{name}" unless names[index]? == name
        reference = native_stages[index]
        unless rows == reference.rows && channels == reference.channels && actual.count == reference.values.count
          raise "coalesced stage shape mismatch #{name}"
        end
        context.dispatch("dr3_dac_compare_stage", SOURCE, [actual, reference.values, receipt],
          [actual.count.to_u64, index.to_u64], actual.count)
        observed << name
        nil
      }
      decoder.decode(codes, frames, false)
      decoder.device_trace = nil
      raise "incomplete coalesced stage capture" unless observed == names
      raise "coalesced decode read back activations" unless context.readbacks.empty?
      words = read_compact!(context, receipt, STAGE_COUNT * 2, "terminal")
      results = names.each_with_index.map do |name, index|
        {name: name, rows: native_stages[index].rows, channels: native_stages[index].channels,
         bit_mismatches: words[index * 2], nonfinite_pairs: words[index * 2 + 1]}
      end
      results.to_a
    end

    # This path reads only the compact integer witness after a known terminal
    # drain. It does not call Context#read or trigger another queue operation.
    def self.read_compact!(context : Context, receipt : Buffer, words : Int32, purpose : String) : Array(UInt32)
      before = context.graph_stats
      raise "compact receipt read before terminal queue" unless before["pending"] == 0 && before["completed"] == before["submitted"]
      pointer = receipt.buffer.contents.as(UInt32*) + receipt.offset // 4
      result = Array(UInt32).new(words) { |index| pointer[index] }
      context.readbacks[purpose] += words.to_i64 * 4_i64
      raise "compact receipt read changed queue state" unless context.graph_stats == before
      result
    end

    def self.run_negative_control!(context : Context) : NamedTuple(expected_bit_mismatches: UInt32, expected_nonfinite_pairs: UInt32, observed_bit_mismatches: UInt32, observed_nonfinite_pairs: UInt32, diagnostic_submissions: Int64, diagnostic_waits: Int64)
      actual = context.upload([0x3f800001_u32.unsafe_as(Float32), 0x7fc00000_u32.unsafe_as(Float32)])
      reference = context.upload([1.0_f32, 0.0_f32])
      receipt = context.upload([0.0_f32, 0.0_f32])
      before = context.graph_stats
      context.dispatch("dr3_dac_compare_stage", SOURCE, [actual, reference, receipt], [2_u64, 0_u64], 2)
      context.finish
      after = context.graph_stats
      diagnostic_submissions = after["submitted"] - before["submitted"]
      diagnostic_waits = after["completion_waits"] - before["completion_waits"]
      unless diagnostic_submissions == 1 && after["completed"] == before["completed"] + 1 && diagnostic_waits == 1 && after["pending"] == 0
        raise "negative-control dispatch did not drain as one isolated command"
      end
      words = read_compact!(context, receipt, 2, "control")
      raise "GPU comparison negative control did not detect both bit differences and nonfinite input" unless words == [2_u32, 1_u32]
      {expected_bit_mismatches: 2_u32, expected_nonfinite_pairs: 1_u32,
       observed_bit_mismatches: words[0], observed_nonfinite_pairs: words[1],
       diagnostic_submissions: diagnostic_submissions,
       diagnostic_waits: diagnostic_waits}
    end

    def self.median(values : Array(Float64)) : Float64
      sorted = values.sort
      raise "median requires samples" if sorted.empty?
      middle = sorted.size // 2
      sorted.size.odd? ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0
    end

    def self.run : Nil
      unless ARGV.size.in?(1..2) && (ARGV.size == 1 || ARGV[1] == "--timings=8")
        raise "usage: drama3_dac_layout_bench MODEL_DIR [--timings=8]"
      end
      model_dir = File.expand_path(ARGV[0])
      fixture = load_fixture
      codec_path = File.join(model_dir, "codec.safetensors")
      codec_sha = sha_file(codec_path)
      raise "codec checkpoint digest mismatch: #{codec_sha}" unless codec_sha == CODEC_SHA256

      configure_environment!
      context = Context.new
      raise "CogniGraph depth must be 2" unless context.graph_stats["depth"] == 2
      store = WeightStore.new(model_dir)
      native = new_decoder(store, context, "native")
      coalesced = new_decoder(store, context, "coalesced")
      code_buffer = context.upload(fixture[:codes])

      # Unscored primes warm the converted checkpoint tensors in one fixed store.
      native.decode(code_buffer, fixture[:frames], false)
      coalesced.decode(code_buffer, fixture[:frames], false)
      raise "prime decode readbacks" unless context.readbacks.empty?

      timing = [] of Timing
      TIMING_ORDER.each_with_index do |layout, index|
        decoder = layout == "native" ? native : coalesced
        timing << measure_decode(context, decoder, code_buffer, fixture[:frames], index.to_i32, layout)
      end

      names = expected_names
      native_stages = capture_native(context, native, code_buffer, fixture[:frames], names)
      comparison = compare_coalesced(context, coalesced, code_buffer, fixture[:frames], native_stages, names)
      raise "missing materialized DAC stage results" unless comparison.size == STAGE_COUNT
      raise "unexpected readback purposes after matched compare" unless context.readbacks.keys == ["terminal"]
      negative = run_negative_control!(context)
      raise "readback ledger mismatch" unless context.readbacks["terminal"] == STAGE_COUNT * 2 * 4 && context.readbacks["control"] == 2 * 4

      durations = {"native" => timing.select { |sample| sample.layout == "native" }.map(&.seconds),
                   "coalesced" => timing.select { |sample| sample.layout == "coalesced" }.map(&.seconds)}
      medians = durations.transform_values { |values| median(values) }
      speech_seconds = fixture[:frames].to_f64 * 2048.0 / 44_100.0
      bit_mismatches = comparison.sum(&.[:bit_mismatches].to_u64)
      nonfinite_pairs = comparison.sum(&.[:nonfinite_pairs].to_u64)
      report = {
        format: "drama3-dac-layout-benchmark-v1",
        fixture: {path: FIXTURE_PATH, sha256: FIXTURE_SHA256, frames: fixture[:frames],
                  frame_layout: "[main,c0..c9]", upload_layout: "frame_major_f32", upload_sha256: fixture[:codes_sha256]},
        codec: {path: codec_path, safetensors_sha256: codec_sha},
        sources: {
          benchmark_cr_sha256: sha_file(File.join(__DIR__, "drama3_dac_layout_bench.cr")),
          compare_metal_compiled_sha256: Digest::SHA256.hexdigest(SOURCE),
          compare_metal_file_sha256: sha_file(File.join(__DIR__, "support/drama3_dac_layout_compare.metal")),
          codec_resident_cr_sha256: sha_file(File.join(__DIR__, "../src/ml/audio/drama3/codec_resident.cr")),
          codec_resident_metal_sha256: sha_file(File.join(__DIR__, "../src/ml/audio/drama3/kernels/codec_resident.metal")),
        },
        environment: {backend: "metal", graph_depth: 2, dac_scratch: false, profilers: false,
                      timing_order: TIMING_ORDER, expected_commands_per_decode: EXPECTED_COMMANDS_PER_DECODE},
        timing: {samples_seconds: durations, medians_seconds: medians,
                 speech_seconds: speech_seconds,
                 rtf: {native: medians["native"] / speech_seconds, coalesced: medians["coalesced"] / speech_seconds},
                 per_decode_graph_deltas: timing.map { |sample|
                   {order_index: sample.order_index, layout: sample.layout, seconds: sample.seconds,
                    submitted: sample.submitted, completed: sample.completed, completion_waits: sample.waits}
                 }},
        matched_comparison: {reference_layout: "native", actual_layout: "coalesced",
                             comparison: "exact_f32_bits; nonfinite pairs counted independently",
                             stages: comparison, total_bit_mismatches: bit_mismatches,
                             total_nonfinite_pairs: nonfinite_pairs,
                             exact_all_stage_bits: bit_mismatches == 0 && nonfinite_pairs == 0},
        negative_control: negative,
        readback_bytes: {timed_arms: 0, diagnostic_activations: 0,
                         terminal_compact_receipt: context.readbacks["terminal"],
                         control_compact_receipt: context.readbacks["control"]},
        final_graph: context.graph_stats,
      }
      puts report.to_json
      unless bit_mismatches == 0 && nonfinite_pairs == 0
        raise "native/coalesced DAC stage bit comparison failed; see full report above"
      end
    end
  end

  Drama3DacLayoutBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
