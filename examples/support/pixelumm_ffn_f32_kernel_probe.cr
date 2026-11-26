require "json"
require "digest/sha256"
require "c/unistd"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_ffn_isolated_probe"
require "./pixelumm_ffn_projection_probe"
require "./pixelumm_memory_probe"
require "./hy_nomas_probe_instrumentation"

module PixelUMMFFNF32KernelProbe
  extend self

  SCHEMA            = "cogni_ml.pixelumm_ffn_f32_kernel_probe/v1"
  NATIVE_SOURCE     = {{ read_file("#{__DIR__}/pixelumm_ffn_f32_gemm.metal") }}
  COMPARATOR_SOURCE = PixelUMMFFNIsolatedProbe::SOURCE
  TM                =  64
  TN                =  32
  TK                =  32
  THREADS           = 128
  BASE_KERNEL       = "pixelumm_ffn_f32_b64x32"
  REUSE_B_KERNEL    = "pixelumm_ffn_f32_b64x32_reuse_b"
  METRIC_SLOTS      = 11 # 2 warmups, 8 timed requests, 1 injected-NaN control.
  REQUESTS          = 10
  GIB               = 1024_i64 * 1024 * 1024
  MAX_NAMED_BYTES   = 4_i64 * GIB
  alias ScheduleEntry = NamedTuple(index: Int32, arm: String, warmup: Bool)

  def projection(name : String) : NamedTuple(input: String, output: String, weight: String, k: Int32, n: Int32)
    PixelUMMFFNProjectionProbe.projection(name)
  end

  def resources(name : String, rows : Int32) : Hash(String, Int64)
    raise ArgumentError.new("unsupported PixelUMM FFN row count") unless {2048, 21120}.includes?(rows)
    shape = projection(name)
    input_bytes = rows.to_i64 * shape[:k] * 2
    weight_bytes = shape[:k].to_i64 * shape[:n] * 2
    output_bytes = rows.to_i64 * shape[:n] * 2
    {
      "input"    => input_bytes,
      "weight"   => weight_bytes,
      "expected" => output_bytes,
      "output"   => output_bytes,
      "metrics"  => METRIC_SLOTS.to_i64 * 4 * sizeof(UInt32),
    }
  end

  def named_bytes(name : String, rows : Int32) : Int64
    resources(name, rows).values.sum(0_i64)
  end

  # Called before fixture metadata/payload is read and before Metal initializes.
  def validate_budget!(name : String, rows : Int32, maximum : Int64) : Int64
    unless maximum > 0 && maximum <= MAX_NAMED_BYTES
      raise ArgumentError.new("PixelUMM FFN exact named-memory cap must be between one byte and 4 GiB")
    end
    required = named_bytes(name, rows)
    raise ArgumentError.new("selected PixelUMM FFN buffers exceed cap before fixture read/Metal initialization") if required > maximum
    required
  end

  def geometry(rows : Int32, k : Int32, n : Int32) : NamedTuple(groups: Tuple(Int32, Int32, Int32), threads: Tuple(Int32, Int32, Int32), output_elements: Int64)
    unless rows > 0 && k > 0 && n > 0 && rows % TM == 0 && k % TK == 0 && n % TN == 0
      raise ArgumentError.new("F32 FFN kernel requires full 64x32x32 tiles")
    end
    {groups: {n // TN, rows // TM, 1}, threads: {THREADS, 1, 1},
     output_elements: rows.to_i64 * n}
  end

  # Reverse warmup order and measured order independently for the paired cell.
  # Each arm has four measured requests in either direction.
  def schedule(reverse_order : Bool) : Array(ScheduleEntry)
    warmups = ["mps.f32", "native.f32"]
    measured = ["mps.f32", "native.f32", "native.f32", "mps.f32",
                "native.f32", "mps.f32", "mps.f32", "native.f32"]
    if reverse_order
      warmups = warmups.reverse
      measured = measured.reverse
    end
    entries = [] of ScheduleEntry
    warmups.each_with_index { |arm, index| entries << {index: index.to_i32, arm: arm, warmup: true} }
    measured.each_with_index { |arm, index| entries << {index: (index + 2).to_i32, arm: arm, warmup: false} }
    entries
  end

  def median(values : Array(Float64)) : Float64
    PixelUMMFFNProjectionProbe.median(values)
  end

  # Isolation requires valid intervals for outside roots as well as the timed
  # projection. Unknown outside intervals must not silently imply zero overlap.
  def validate_root_timings!(roots : Array(ML::Video::MPSCommandRootTiming)) : Nil
    raise ArgumentError.new("F32 FFN root timings are empty") if roots.empty?
    roots.each_with_index do |root, index|
      unless root.sequence == index && root.completed && !root.has_error &&
             root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
             root.gpu_start_seconds > 0 && root.gpu_end_seconds > root.gpu_start_seconds
        raise ArgumentError.new("F32 FFN timing root #{index} is incomplete, invalid or unmeasured")
      end
    end
  end

  {% unless flag?(:cpu_only) %}
    def checked_root_timing(timing : PixelUMMFFNIsolatedProbe::StageTiming,
                            roots : Array(ML::Video::MPSCommandRootTiming))
      validate_root_timings!(roots)
      PixelUMMFFNIsolatedProbe.root_timing(timing, roots)
    end

    class Owner
      getter resources : Hash(String, Int64)
      getter named_bytes : Int64
      getter input_record : PixelUMMFFNIsolatedProbe::TensorRecord
      getter weight_record : PixelUMMFFNIsolatedProbe::TensorRecord
      getter expected_record : PixelUMMFFNIsolatedProbe::TensorRecord
      getter lifecycle : PixelUMMFFNIsolatedProbe::SubmissionLifecycle
      getter rows : Int32
      getter k : Int32
      getter n : Int32
      getter product_type : UInt32 = 0_u32
      getter native_static_threadgroup_bytes : Int64 = 0_i64
      getter native_execution_width : Int32 = 0
      getter native_max_threads : Int32 = 0
      getter kernel_name : String

      @buffers = {} of String => ML::MetalBuffer
      @native_pipeline : ML::Metal::ComputePipeline? = nil
      @compare_pipeline : ML::Metal::ComputePipeline? = nil
      @plan = Pointer(Void).null
      @ffi_error : Bytes
      @released = false

      def initialize(@fixture : PixelUMMFFNIsolatedProbe::Fixture, name : String,
                     @rows : Int32, maximum : Int64,
                     @memory : PixelUMMMemoryProbe::Recorder,
                     @kernel_name : String)
        selected = PixelUMMFFNF32KernelProbe.projection(name)
        @k, @n = selected[:k], selected[:n]
        @input_record = @fixture.metadata.inputs[selected[:input]]
        @weight_record = @fixture.metadata.weights[selected[:weight]]
        @expected_record = @fixture.metadata.captures[selected[:output]]
        unless @fixture.metadata.tokens == @rows &&
               @input_record.shape == [1, @rows, @k] &&
               @weight_record.shape == [@n, @k] &&
               @expected_record.shape == [1, @rows, @n] &&
               {@input_record, @weight_record, @expected_record}.all? { |record| record.dtype == "BF16" }
          raise ArgumentError.new("selected PixelUMM FFN fixture records do not match the exact projection")
        end

        @resources = PixelUMMFFNF32KernelProbe.resources(name, @rows)
        @named_bytes = PixelUMMFFNF32KernelProbe.validate_budget!(name, @rows, maximum)
        @lifecycle = PixelUMMFFNIsolatedProbe::SubmissionLifecycle.new
        @ffi_error = Bytes.new(1024, 0_u8)
        raise "Metal unavailable" unless ML::Metal::Device.init!
        @memory.emit("prebuffers", phase_bytes_kind: "planned_named_metal_bytes", phase_bytes: @named_bytes)
        begin
          @resources.each { |key, bytes| @buffers[key] = ML::MetalBuffer.new(bytes) }
          @fixture.read_into(@input_record, buffer("input"))
          @fixture.read_into(@weight_record, buffer("weight"))
          @fixture.read_into(@expected_record, buffer("expected"))
          {"output", "metrics"}.each do |key|
            Slice.new(buffer(key).contents.as(UInt8*), buffer(key).size.to_i32).fill(0_u8)
          end
          @memory.emit("postupload", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)

          @native_pipeline = ML::Metal::ComputePipeline.new(@kernel_name, PixelUMMFFNF32KernelProbe::NATIVE_SOURCE)
          @compare_pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16", PixelUMMFFNF32KernelProbe::COMPARATOR_SOURCE)
          @native_static_threadgroup_bytes = @native_pipeline.not_nil!.static_threadgroup_memory_length
          @native_execution_width = @native_pipeline.not_nil!.thread_execution_width
          @native_max_threads = @native_pipeline.not_nil!.max_total_threads_per_threadgroup
          unless @native_execution_width == 32 && @native_max_threads >= PixelUMMFFNF32KernelProbe::THREADS &&
                 @native_static_threadgroup_bytes == 12_288
            raise ArgumentError.new("native F32 pipeline does not meet the fixed four-simdgroup/12-KiB tile contract")
          end

          error = @ffi_error
          type = 0_u32
          selected_plan = PixelUMMProjectionFFI.create(
            ML::Metal::Device.instance.device_handle,
            buffer("input").handle, buffer("weight").handle, buffer("output").handle,
            @rows, @k, @n, 1, pointerof(type), error.to_unsafe, error.size)
          raise "MPS F32 projection preparation: #{String.new(error.to_unsafe)}" if selected_plan.null?
          @plan = selected_plan
          @product_type = type
          @memory.emit("postcompile:mps_and_native", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def buffer(name : String) : ML::MetalBuffer
        @buffers[name]
      end

      def preencode! : Nil
        @memory.emit("preencode", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def poison!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
        raise "F32 probe output poison blit unavailable" if blit.null?
        MetalDispatchFFI.blit_fill_buffer(blit, buffer("output").handle, 255_u8, 0_i64, buffer("output").size)
        MetalDispatchFFI.blit_end_encoding(blit)
        chain.resume!
      end

      def encode!(chain : ML::Metal::MPSCommandChain, arm : String) : Nil
        @lifecycle.mark_attempt!
        case arm
        when "mps.f32"
          chain.pause!
          status = PixelUMMProjectionFFI.encode(@plan, chain.handle, @ffi_error.to_unsafe, @ffi_error.size)
          chain.resume!
          raise "MPS F32 projection encode #{status}: #{String.new(@ffi_error.to_unsafe)}" unless status == 0
        when "native.f32"
          pipeline = @native_pipeline || raise ArgumentError.new("native F32 pipeline is unavailable")
          shape = PixelUMMFFNF32KernelProbe.geometry(@rows, @k, @n)
          chain.encoder.set_pipeline(pipeline)
            .set_buffer(buffer("weight"), 0)
            .set_buffer(buffer("input"), 1)
            .set_buffer(buffer("output"), 2, ML::Metal::BufferAccess::Write)
            .set_value(@k.to_u32, 3)
            .set_value(@n.to_u32, 4)
            .set_value(@rows.to_u32, 5)
            .dispatch_threadgroups(shape[:groups], shape[:threads])
        else
          raise ArgumentError.new("unknown F32 FFN arm: #{arm}")
        end
      end

      def compare!(chain : ML::Metal::MPSCommandChain, sample : Int32) : Nil
        pipeline = @compare_pipeline || raise ArgumentError.new("BF16 comparator pipeline is unavailable")
        @lifecycle.mark_attempt!
        count = PixelUMMFFNF32KernelProbe.geometry(@rows, @k, @n)[:output_elements].to_u32
        chain.encoder.set_pipeline(pipeline)
          .set_buffer(buffer("output"), 0)
          .set_buffer(buffer("expected"), 1)
          .set_buffer(buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite, sample.to_i64 * 16)
          .set_value(count, 3)
          .set_value(count, 4)
          .set_value(0_u32, 5)
          .dispatch_threadgroups({(count.to_i32 + 255) // 256, 1, 1}, {256, 1, 1})
      end

      def positive_control!(chain : ML::Metal::MPSCommandChain) : Nil
        pipeline = @compare_pipeline || raise ArgumentError.new("BF16 comparator pipeline is unavailable")
        @lifecycle.mark_attempt!
        count = PixelUMMFFNF32KernelProbe.geometry(@rows, @k, @n)[:output_elements].to_u32
        chain.encoder.set_pipeline(pipeline)
          .set_buffer(buffer("expected"), 0)
          .set_buffer(buffer("expected"), 1)
          .set_buffer(buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite, PixelUMMFFNF32KernelProbe::REQUESTS.to_i64 * 16)
          .set_value(count, 3)
          .set_value(count, 4)
          .set_value(1_u32, 5)
          .dispatch_threadgroups({(count.to_i32 + 255) // 256, 1, 1}, {256, 1, 1})
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(PixelUMMFFNIsolatedProbe::Metric)
        unless chain.completed? && chain.wait_count == 1
          raise ArgumentError.new("F32 probe metric read requires the sole successful terminal wait")
        end
        words = Array(UInt32).new(PixelUMMFFNF32KernelProbe::METRIC_SLOTS * 4, 0_u32)
        buffer("metrics").read_bytes(words.to_unsafe.as(UInt8*), words.size * sizeof(UInt32))
        Array.new(PixelUMMFFNF32KernelProbe::METRIC_SLOTS) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def postwait!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_terminal_wait!(chain.completed?, chain.wait_count)
        @memory.emit("postwait", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
        unless @lifecycle.release_allowed?(chain.completed?, chain.wait_count)
          raise ArgumentError.new("F32 probe buffers/plan cannot be released before successful terminal drain")
        end
        release_owned!
        @memory.emit("after_all_buffers_release", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
      end

      def release_unsubmitted! : Nil
        return if @released
        unless @lifecycle.release_allowed?(false, 0)
          raise ArgumentError.new("F32 probe objects cannot be released after a submission attempt")
        end
        release_owned!
      end

      private def release_owned! : Nil
        return if @released
        unless @plan.null?
          PixelUMMProjectionFFI.release(@plan)
          @plan = Pointer(Void).null
        end
        @buffers.each_value(&.release)
        @buffers.clear
        @released = true
      end
    end

    # Retain objects for process lifetime if a submitted command cannot be
    # successfully terminally drained; their finalizers must not release GPU data.
    class FailureQuarantine
      @@owners = [] of Owner
      @@chains = [] of ML::Metal::MPSCommandChain

      def self.retain(owner : Owner, chain : ML::Metal::MPSCommandChain) : Nil
        @@owners << owner
        @@chains << chain
      end

      def self.retain_chain(chain : ML::Metal::MPSCommandChain) : Nil
        @@chains << chain
      end
    end
  {% end %}

  def metric_json(metric : PixelUMMFFNIsolatedProbe::Metric) : JSON::Any
    JSON.parse({failures: metric[:failures], different_words: metric[:different_words],
                nonfinite: metric[:nonfinite], elements: metric[:elements]}.to_json)
  end

  def write_report_new!(path : String, content : String) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.from_errno("create F32 FFN report") if fd < 0
    file = IO::FileDescriptor.new(fd)
    begin
      file.puts(content)
    ensure
      file.close
    end
  end
end
