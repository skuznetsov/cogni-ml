require "json"
require "digest/sha256"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_ffn_aa_probe"

module PixelUMMFFNGraphProbe
  extend self

  enum Topology
    FullGraph
    SplitDown
    StageOutputs

    def self.parse(value : String) : Topology
      case value
      when "full_graph" then FullGraph
      when "split_down" then SplitDown
      when "stage_outputs" then StageOutputs
      else raise ArgumentError.new("unknown FFN graph topology: #{value}")
      end
    end

    def self.select(split_down : Bool = false, stage_outputs : Bool = false) : Topology
      raise ArgumentError.new("--split-down and --stage-outputs are mutually exclusive") if split_down && stage_outputs
      return StageOutputs if stage_outputs
      split_down ? SplitDown : FullGraph
    end

    def label : String
      case self
      when FullGraph then "full_graph"
      when SplitDown then "split_down"
      when StageOutputs then "stage_outputs"
      else raise "unknown FFN graph topology"
      end
    end

    def split_down? : Bool
      self == SplitDown
    end

    def stage_outputs? : Bool
      self == StageOutputs
    end

    def uses_split_down? : Bool
      self != FullGraph
    end
  end

  alias A = PixelUMMFFNAaProbe
  alias B = PixelUMMFFNWeightReuseProbe
  alias C = PixelUMMFFNAaContract
  alias I = PixelUMMFFNIsolatedProbe

  SCHEMA            = "cogni_ml.pixelumm_ffn_graph_probe/v1"
  GRAPH_OPERATION_NAMES = [
    "output_only_topology_graph_encodes",
    "output_only_split_down_graph_encodes",
    "captured_topology_graph_encodes",
    "captured_split_down_graph_encodes",
    "reference_silu_kernel_encodes",
  ]
  FULL_GRAPH_OPERATION_COUNTS = [1_u64, 0_u64, 1_u64, 0_u64, 1_u64]
  SPLIT_DOWN_OPERATION_COUNTS = [1_u64, 1_u64, 1_u64, 1_u64, 1_u64]
  FFN_CAPTURE_NAMES = ["graph.gate", "graph.up", "graph.silu", "graph.product"]
  STAGE_NAMES       = [
    "output_only.down_vs_saved",
    "output_only.down_vs_native_baseline",
    "output_only.product_vs_saved",
    "output_only.product_vs_native_baseline",
    "output_only.gate_vs_saved",
    "output_only.gate_vs_native_baseline",
    "output_only.up_vs_saved",
    "output_only.up_vs_native_baseline",
    "output_only.silu_vs_native_baseline",
    "baseline.gate_vs_saved",
    "baseline.up_vs_saved",
    "captured.gate_vs_saved",
    "captured.gate_vs_baseline",
    "captured.up_vs_saved",
    "captured.up_vs_baseline",
    "baseline.silu_vs_derived_saved",
    "captured.silu_vs_derived_saved",
    "captured.silu_vs_baseline",
    "baseline.product_vs_saved",
    "captured.product_vs_saved",
    "captured.product_vs_baseline",
    "baseline.down_vs_saved",
    "captured.down_vs_saved",
    "captured.down_vs_baseline",
    "positive_control.injected_nan",
  ]
  STAGE_COUNTS = [
    C::DOWN_ELEMENTS, C::DOWN_ELEMENTS,
    C::PRODUCT_ELEMENTS, C::PRODUCT_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS, C::RAW_FFN_ELEMENTS,
    C::PRODUCT_ELEMENTS, C::PRODUCT_ELEMENTS, C::PRODUCT_ELEMENTS,
    C::DOWN_ELEMENTS, C::DOWN_ELEMENTS, C::DOWN_ELEMENTS, C::DOWN_ELEMENTS,
  ]
  alias Metric = NamedTuple(failures: UInt32, different_words: UInt32,
    nonfinite: UInt32, elements: UInt32)

  def sha_file(path : String) : String
    hash = Digest::SHA256.new
    File.open(path) do |io|
      chunk = Bytes.new(1024 * 1024)
      loop do
        count = io.read(chunk)
        break if count == 0
        hash.update(chunk[0, count])
      end
    end
    hash.hexfinal
  end

  def graph_resource_bytes : Hash(String, Int64)
    wide = C::TOKENS.to_i64 * C::FFN * 2_i64
    FFN_CAPTURE_NAMES.to_h { |name| {name, wide} }
  end

  def expected_candidate_counts(topology : Topology) : Array(UInt64)
    topology.uses_split_down? ? SPLIT_DOWN_OPERATION_COUNTS : FULL_GRAPH_OPERATION_COUNTS
  end

  def validate_candidate_counts!(topology : Topology,
                                 counts : Array(UInt64)) : Array(String)
    raise ArgumentError.new("FFN graph probe operation count mismatch") unless counts.size == GRAPH_OPERATION_NAMES.size
    expected = expected_candidate_counts(topology)
    failures = [] of String
    GRAPH_OPERATION_NAMES.each_with_index do |name, index|
      next if counts[index] == expected[index]
      failures << "#{name}: observed #{counts[index]} expected #{expected[index]} for #{topology.label}"
    end
    failures
  end

  def operation_count_report(counts : Array(UInt64)) : Hash(String, UInt64)
    raise ArgumentError.new("FFN graph probe operation count mismatch") unless counts.size == GRAPH_OPERATION_NAMES.size
    GRAPH_OPERATION_NAMES.each_with_index.to_h { |name, index| {name, counts[index]} }
  end

  def expected_stage_counts(topology : Topology) : Array(Int32)
    counts = STAGE_COUNTS.dup
    case topology
    when Topology::FullGraph
      (2..8).each { |slot| counts[slot] = 0 }
    when Topology::SplitDown
      (4..8).each { |slot| counts[slot] = 0 }
    when Topology::StageOutputs
      # All four BF16 front-stage outputs are mandatory graph targets.
    end
    counts
  end

  def validate_metrics!(metrics : Array(Metric), topology : Topology = Topology::FullGraph) : Array(String)
    raise ArgumentError.new("FFN graph probe metric count mismatch") unless metrics.size == STAGE_NAMES.size
    failures = [] of String
    expected_counts = expected_stage_counts(topology)
    (0...STAGE_NAMES.size).each do |slot|
      metric = metrics[slot]
      count = expected_counts[slot].to_u32
      if slot == STAGE_NAMES.size - 1
        unless metric == {failures: 1_u32, different_words: 0_u32,
                          nonfinite: 1_u32, elements: count}
          failures << "#{STAGE_NAMES[slot]}: expected the exact injected-NaN control"
        end
      elsif count == 0
        unless metric == {failures: 0_u32, different_words: 0_u32,
                          nonfinite: 0_u32, elements: 0_u32}
          failures << "#{STAGE_NAMES[slot]}: inactive metric must be exactly zero, observed #{metric}"
        end
      elsif metric[:elements] != count || metric[:failures] != 0 || metric[:nonfinite] != 0
        failures << "#{STAGE_NAMES[slot]}: failed/inactive metric #{metric} expected_elements=#{count}"
      end
    end
    failures
  end

  {% unless flag?(:cpu_only) %}
    lib FFNGraphProbeFFI
      fun create = gs_pixelumm_ffn_graph_probe_create(
        device : Void*, norm_input : Void*, gate_weight : Void*, up_weight : Void*,
        down_weight : Void*, gate_output : Void*, up_output : Void*, silu_output : Void*,
        product_output : Void*, down_output : Void*, tokens : Int32,
        topology : Int32,
        error : UInt8*, capacity : LibC::SizeT,
      ) : Void*
      fun encode_output_only = gs_pixelumm_ffn_graph_probe_encode_output_only(
        plan : Void*, command : Void*, error : UInt8*, capacity : LibC::SizeT,
      ) : Int32
      fun encode_captured = gs_pixelumm_ffn_graph_probe_encode_captured(
        plan : Void*, command : Void*, error : UInt8*, capacity : LibC::SizeT,
      ) : Int32
      fun derive_reference_silu = gs_pixelumm_ffn_graph_probe_derive_reference_silu(
        plan : Void*, command : Void*, expected_gate : Void*,
        error : UInt8*, capacity : LibC::SizeT,
      ) : Int32
      fun stats = gs_pixelumm_ffn_graph_probe_stats(
        plan : Void*, counts : UInt64*, count : Int32,
      ) : Int32
      fun release = gs_pixelumm_ffn_graph_probe_release(plan : Void*) : Void
    end

    class Owner
      getter lifecycle = I::SubmissionLifecycle.new
      getter named_bytes : Int64
      getter resources : Hash(String, Int64)
      getter topology : Topology

      @baseline : A::Owner
      @buffers = {} of String => ML::MetalBuffer
      @plan : Pointer(Void) = Pointer(Void).null
      @pipeline : ML::Metal::ComputePipeline? = nil
      @error_text = Bytes.new(1024, 0_u8)
      @released = false

      def initialize(@baseline : A::Owner, @topology : Topology = Topology::FullGraph)
        @resources = @baseline.resources.merge(PixelUMMFFNGraphProbe.graph_resource_bytes)
        @named_bytes = @resources.values.sum(0_i64)
        raise ArgumentError.new("whole-FFN graph named-memory cap exceeded") unless @named_bytes <= A::MAX_NAMED_BYTES
        raise "Metal unavailable" unless ML::Metal::Device.init!
        begin
          PixelUMMFFNGraphProbe.graph_resource_bytes.each do |name, bytes|
            buffer = ML::MetalBuffer.new(bytes)
            @buffers[name] = buffer
            Slice.new(buffer.contents.as(UInt8*), buffer.size.to_i32).fill(255_u8)
          end
          @plan = FFNGraphProbeFFI.create(
            ML::Metal::Device.instance.device_handle,
            buffer("input.norm").handle,
            buffer("weight.gate").handle,
            buffer("weight.up").handle,
            buffer("weight.down").handle,
            buffer("graph.gate").handle,
            buffer("graph.up").handle,
            buffer("graph.silu").handle,
            buffer("graph.product").handle,
            buffer("work.down").handle,
            C::TOKENS, @topology.value,
            @error_text.to_unsafe, @error_text.size)
          raise "whole-FFN graph preparation: #{String.new(@error_text.to_unsafe)}" if @plan.null?
          @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16", A::SOURCE)
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def buffer(name : String) : ML::MetalBuffer
        @buffers[name]? || @baseline.buffer(name)
      end

      def encode_output_only!(chain : ML::Metal::MPSCommandChain) : Nil
        encode_graph!(chain) do |command|
          FFNGraphProbeFFI.encode_output_only(@plan, command,
            @error_text.to_unsafe, @error_text.size)
        end
      end

      def encode_captured!(chain : ML::Metal::MPSCommandChain) : Nil
        encode_graph!(chain) do |command|
          FFNGraphProbeFFI.encode_captured(@plan, command,
            @error_text.to_unsafe, @error_text.size)
        end
      end

      # The source fixture stores gate_proj but has no silu_gate capture. Derive
      # that one reference stage in place with the exact production kernel after
      # all raw gate comparisons have been enqueued.
      def derive_reference_silu!(chain : ML::Metal::MPSCommandChain) : Nil
        encode_graph!(chain) do |command|
          FFNGraphProbeFFI.derive_reference_silu(@plan, command,
            @baseline.buffer("expected.gate").handle,
            @error_text.to_unsafe, @error_text.size)
        end
      end

      def compare!(chain : ML::Metal::MPSCommandChain, actual : String,
                   expected : String, slot : Int32, count : Int32,
                   inject = false) : Nil
        raise ArgumentError.new("FFN graph comparator slot is outside its metric vector") unless (0...STAGE_NAMES.size).includes?(slot)
        @lifecycle.mark_attempt!
        chain.encoder.set_pipeline(@pipeline || raise ArgumentError.new("FFN graph comparator unavailable"))
          .set_buffer(buffer(actual), 0)
          .set_buffer(buffer(expected), 1)
          .set_buffer(@baseline.buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite,
            slot.to_i64 * 4_i64 * sizeof(UInt32))
          .set_value(count.to_u32, 3)
          .set_value(count.to_u32, 4)
          .set_value(inject ? 1_u32 : 0_u32, 5)
          .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
      end

      def positive_control!(chain : ML::Metal::MPSCommandChain) : Nil
        compare!(chain, "canonical.down", "canonical.down",
          (STAGE_NAMES.size - 1).to_i32, C::DOWN_ELEMENTS, inject: true)
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(Metric)
        raise "FFN graph metrics require the sole terminal wait" unless chain.completed? && chain.wait_count == 1
        words = Array(UInt32).new(STAGE_NAMES.size * 4, 0_u32)
        @baseline.buffer("metrics").read_bytes(words.to_unsafe.as(UInt8*), words.size * sizeof(UInt32))
        Array.new(STAGE_NAMES.size) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def counts : Array(UInt64)
        values = Array(UInt64).new(PixelUMMFFNGraphProbe::GRAPH_OPERATION_NAMES.size, 0_u64)
        raise "whole-FFN graph stats failed" unless FFNGraphProbeFFI.stats(@plan, values.to_unsafe, values.size) == 0
        values
      end

      def operation_count_report : Hash(String, UInt64)
        PixelUMMFFNGraphProbe.operation_count_report(counts)
      end

      def postwait!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_terminal_wait!(chain.completed?, chain.wait_count)
      end

      def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
        raise "whole-FFN graph unsafe release" unless @lifecycle.release_allowed?(chain.completed?, chain.wait_count)
        release_resources!
      end

      def release_unsubmitted! : Nil
        return if @released
        raise "whole-FFN graph submitted resources require terminal wait" unless @lifecycle.release_allowed?(false, 0)
        release_resources!
      end

      private def encode_graph!(chain : ML::Metal::MPSCommandChain, &block : Pointer(Void) -> Int32) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        status = block.call(chain.handle)
        chain.resume!
        raise "whole-FFN graph encode: #{status}: #{String.new(@error_text.to_unsafe)}" unless status == 0
      end

      private def release_resources! : Nil
        return if @released
        FFNGraphProbeFFI.release(@plan) unless @plan.null?
        @plan = Pointer(Void).null
        @buffers.each_value(&.release)
        @buffers.clear
        @released = true
      end
    end
  {% end %}
end
