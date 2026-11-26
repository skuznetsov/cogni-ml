# Bounded Metal diagnostic for whether a file-backed buffer's exposed extent
# changes first-submit scheduling time. This does not identify pinning versus
# page faults, nor model-specific deferred work.
#
# Build:
#   make build/bridge.o
#   crystal build --release -Ddrama3_metal \
#     --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld" \
#     examples/drama3_startup_probe.cr -o /tmp/drama3_startup_probe
#
# Run each extent in a fresh process under the repository guard:
#   DRAMA3_BACKEND=metal COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
#     scripts/run_safe.sh /tmp/drama3_startup_probe 180 8192 ./model-00001.safetensors --extent=window
#   DRAMA3_BACKEND=metal COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
#     scripts/run_safe.sh /tmp/drama3_startup_probe 180 8192 ./model-00001.safetensors --extent=shard
# Add --prefault to CPU-touch each mapped page before the first command, or
# --madvise-random to apply MADV_RANDOM. Both are explicit interventions and
# their costs are included in full_elapsed_seconds. Set DRAMA3_MMAP_SHARED=1
# to compare the read-only shared mapping experiment; default 0 is private.
# Keep checkpoint files immutable throughout each process.
#
# The window/shard arms use one identical pipeline and exactly five dispatches
# in each of three synchronously completed commands. The internal timer omits
# launch/argument validation and final release; use /usr/bin/time for full
# process timing. Fresh processes do not imply cold OS caches. Interpret any observed
# extent effect only for this tiny kernel and the measured host/device state.

require "../src/ml/audio/drama3/native"
require "../src/ml/audio/drama3/metal"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  usage = "usage: drama3_startup_probe SHARD.safetensors --extent=window|shard [--prefault] [--madvise-random]"
  abort usage if ARGV.size < 2

  shard_path = File.expand_path(ARGV[0])
  abort "shard is not a regular file: #{shard_path}" unless File.file?(shard_path)
  abort "expected one .safetensors shard path" unless File.extname(shard_path) == ".safetensors"

  extent : String? = nil
  cpu_prefault = false
  madvise_random = false
  ARGV.each_with_index do |argument, index|
    next if index == 0

    case argument
    when "--extent=window"
      abort "duplicate --extent option" if extent
      extent = "window"
    when "--extent=shard"
      abort "duplicate --extent option" if extent
      extent = "shard"
    when "--prefault"
      abort "duplicate --prefault option" if cpu_prefault
      cpu_prefault = true
    when "--madvise-random"
      abort "duplicate --madvise-random option" if madvise_random
      madvise_random = true
    else
      abort "unknown option: #{argument}\n#{usage}"
    end
  end
  selected_extent = extent || abort("missing required --extent=window|shard\n#{usage}")

  abort "set DRAMA3_BACKEND=metal" unless ENV["DRAMA3_BACKEND"]? == "metal"

  started = Time.instant
  owner = nil.as(MappedFile?)
  input_buffer = nil.as(ML::MetalBuffer?)
  output_buffer = nil.as(ML::MetalBuffer?)
  last_command = nil.as(ML::Metal::CommandBuffer?)

  begin
    abort "DRAMA-3 Metal backend is unavailable" unless MetalBackend.enabled?

    pipeline_started = Time.instant
    pipeline = ML::Metal::ComputePipeline.new("startup_read", <<-METAL)
      #include <metal_stdlib>
      using namespace metal;
      kernel void startup_read(device const uint *input [[buffer(0)]], device float *output [[buffer(1)]], uint id [[thread_position_in_grid]]) {
        if (id == 0) output[0] = float(input[0]);
      }
      METAL
    pipeline_seconds = (Time.instant - pipeline_started).total_seconds

    page_bytes = MetalBackend::Platform.getpagesize.to_i64
    raise "invalid system page size: #{page_bytes}" unless page_bytes > 0

    setup_started = Time.instant
    mapping_started = Time.instant
    owner = MappedFile.new(shard_path)
    mapping_seconds = (Time.instant - mapping_started).total_seconds
    raise "safetensors shard is too small to contain a 32-bit word" if owner.not_nil!.size < sizeof(UInt32)
    raise "mmap address is not page-aligned" unless owner.not_nil!.pointer.address % page_bytes.to_u64 == 0

    requested_bytes = if selected_extent == "window"
                        page_bytes
                      else
                        ((owner.not_nil!.size + page_bytes - 1) // page_bytes) * page_bytes
                      end
    max_buffer_length = MetalBackend::Platform.gs_max_buffer_length
    raise "requested buffer size #{requested_bytes} exceeds device limit #{max_buffer_length}" if requested_bytes > max_buffer_length

    madvise_seconds = 0.0
    if madvise_random
      madvise_started = Time.instant
      status = LibC.madvise(owner.not_nil!.pointer.as(Void*), owner.not_nil!.size.to_u64, LibC::MADV_RANDOM)
      raise "madvise(MADV_RANDOM) failed" unless status == 0
      madvise_seconds = (Time.instant - madvise_started).total_seconds
    end

    prefault_seconds = 0.0
    prefault_checksum = 0_u64
    if cpu_prefault
      prefault_started = Time.instant
      offset = 0_i64
      while offset < owner.not_nil!.size
        prefault_checksum &+= (owner.not_nil!.pointer + offset).value.to_u64
        offset += page_bytes
      end
      prefault_seconds = (Time.instant - prefault_started).total_seconds
    end

    buffer_started = Time.instant
    input_buffer = ML::MetalBuffer.wrap_no_copy(owner.not_nil!.pointer.as(Void*), requested_bytes)
    input_buffer_seconds = (Time.instant - buffer_started).total_seconds

    output_started = Time.instant
    output_buffer = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    output_buffer_seconds = (Time.instant - output_started).total_seconds
    setup_seconds = (Time.instant - setup_started).total_seconds

    puts({
      event:                  "setup",
      extent:                 selected_extent,
      shard_bytes:            owner.not_nil!.size,
      buffer_bytes:           input_buffer.not_nil!.size,
      page_bytes:             page_bytes,
      dispatches_per_command: 5,
      commands_planned:       3,
      mmap_shared:            ENV["DRAMA3_MMAP_SHARED"]? == "1",
      cpu_prefault:           cpu_prefault,
      prefault_checksum:      cpu_prefault ? prefault_checksum : nil,
      madvise_random:         madvise_random,
      pipeline_seconds:       pipeline_seconds,
      mapping_seconds:        mapping_seconds,
      madvise_seconds:        madvise_seconds,
      prefault_seconds:       prefault_seconds,
      input_buffer_seconds:   input_buffer_seconds,
      output_buffer_seconds:  output_buffer_seconds,
      setup_seconds:          setup_seconds,
      full_elapsed_seconds:   (Time.instant - started).total_seconds,
    }.to_json)

    expected_word = nil.as(UInt32?)
    3.times do |iteration|
      command_started = Time.instant
      command = ML::Metal::CommandBuffer.new
      last_command = command
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoder.set_pipeline(pipeline)
      5.times do
        encoder.set_buffer(input_buffer.not_nil!, 0)
        encoder.set_buffer(output_buffer.not_nil!, 1, ML::Metal::BufferAccess::Write)
        encoder.dispatch_1d(1, 1)
      end
      encoder.end_encoding
      encoded = Time.instant

      gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
      wait_done = Time.instant
      readback_started = Time.instant
      observed = output_buffer.not_nil!.read(1).first
      readback_seconds = (Time.instant - readback_started).total_seconds

      raw_word = expected_word || owner.not_nil!.pointer.as(UInt32*).value
      expected_word = raw_word
      expected = raw_word.to_f32
      raise "GPU result mismatch: observed=#{observed}, expected=float(0x#{raw_word.to_s(16)})=#{expected}" unless observed == expected

      puts({
        event:                "command",
        extent:               selected_extent,
        iteration:            iteration,
        encode_seconds:       (encoded - command_started).total_seconds,
        wait_seconds:         (wait_done - encoded).total_seconds,
        readback_seconds:     readback_seconds,
        gpu_seconds:          gpu_seconds,
        input_first_word:     raw_word,
        output_value:         observed,
        full_elapsed_seconds: (Time.instant - started).total_seconds,
      }.to_json)
    end
  ensure
    begin
      if command = last_command
        command.discard unless command.committed?
      end
    ensure
      begin
        input_buffer.try(&.release)
      ensure
        begin
          output_buffer.try(&.release)
        ensure
          # The file mapping backs the no-copy MTLBuffer; only drop its root
          # after releasing that view, including on command/error paths.
          owner = nil
        end
      end
    end
  end
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
