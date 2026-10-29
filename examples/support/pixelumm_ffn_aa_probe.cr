require "./pixelumm_ffn_aa_contract"
require "./pixelumm_ffn_isolated_probe"
require "./pixelumm_ffn_weight_reuse_probe"

# Diagnostic-only connected FFN. All fixture authentication, allocation and
# compilation happen before the single command-chain sampling loop.
module PixelUMMFFNAaProbe
  extend self
  alias B = PixelUMMFFNWeightReuseProbe
  alias C = PixelUMMFFNAaContract
  alias I = PixelUMMFFNIsolatedProbe
  REFERENCE_SHA256 = "e9b1ddce57fac94efb0262bd3e5f0be2963ef9df4255dc74cf254ff4b0217ddc"
  GIB              = 1024_i64 * 1024 * 1024
  MAX_NAMED_BYTES  = 2_i64 * GIB
  SOURCE           = {{ read_file("#{__DIR__}/pixelumm_ffn_dense.metal") }}

  def resource_bytes : Hash(String, Int64)
    hidden = 5760_i64 * 4096 * 2
    ffn = 5760_i64 * 12288 * 2
    {
      "input.norm" => hidden,
      "weight.gate" => 4096_i64 * 12288 * 2,
      "weight.up" => 4096_i64 * 12288 * 2,
      "weight.down" => 4096_i64 * 12288 * 2,
      "expected.gate" => ffn, "expected.up" => ffn,
      "expected.product" => ffn, "expected.down" => hidden,
      "work.gate" => ffn, "work.up" => ffn, "work.down" => hidden,
      "capture.gate" => ffn, "capture.up" => ffn,
      "canonical.down" => hidden,
      "metrics" => 57_i64 * 4 * sizeof(UInt32),
    }
  end

  {% unless flag?(:cpu_only) %}
    # Serialize only after the existing terminal wait. Keep inclusive graph
    # time and its nested continuation time distinct for the offline analyzer.
    def api_profile_json(profile : ML::Video::MPSAPIProfile) : Array(JSON::Any)
      graph_count = 0_u64
      commit_count = 0_u64
      result = profile.phases.map do |phase|
        operations = {} of String => JSON::Any
        phase.operations.each do |name, stats|
          graph_count += stats.count if name == "graph_encode"
          commit_count += stats.count if name == "commit_and_continue"
          unless (phase.phase_id == 5 || stats.count == 0) &&
                 (name != "t5_linear_encode" || stats.count == 0)
            raise "FFN A/A profile contains unexpected phase/linear work"
          end
          operations[name] = JSON.parse({count: stats.count, wall_total_ns: stats.wall_total_ns,
                                         wall_max_ns: stats.wall_max_ns, thread_cpu_total_ns: stats.thread_cpu_total_ns,
                                         thread_cpu_max_ns: stats.thread_cpu_max_ns, nested_count: stats.nested_count,
                                         nested_wall_total_ns: stats.nested_wall_total_ns,
                                         nested_thread_cpu_total_ns: stats.nested_thread_cpu_total_ns,
                                         exclusive_wall_total_ns: stats.exclusive_wall_total_ns,
                                         exclusive_thread_cpu_total_ns: stats.exclusive_thread_cpu_total_ns}.to_json)
        end
        JSON.parse({phase_id: phase.phase_id, phase: phase.name, operations: operations}.to_json)
      end
      raise "FFN A/A profile changed the fixed command topology" unless graph_count == 54 && commit_count == 161
      result
    end

    lib FFNAaFFI
      fun create = gs_pixelumm_ffn_aa_create(device : Void*, norm_input : Void*,
                                             gate_weight : Void*, up_weight : Void*, down_weight : Void*, gate_work : Void*,
                                             up_work : Void*, down_output : Void*, gate_capture : Void*, up_capture : Void*,
                                             tokens : Int32, error : UInt8*, capacity : LibC::SizeT) : Void*
      fun encode = gs_pixelumm_ffn_aa_encode(plan : Void*, command : Void*,
                                             capture_raw : Int32, error : UInt8*, capacity : LibC::SizeT) : Int32
      fun stats = gs_pixelumm_ffn_aa_stats(plan : Void*, counts : UInt64*, count : Int32) : Int32
      fun release = gs_pixelumm_ffn_aa_release(plan : Void*) : Void
    end

    class Owner
      getter lifecycle = I::SubmissionLifecycle.new
      getter named_bytes : Int64
      getter resources : Hash(String, Int64)
      @buffers = {} of String => ML::MetalBuffer
      @plan : Pointer(Void) = Pointer(Void).null
      @pipeline : ML::Metal::ComputePipeline? = nil
      @error_text = Bytes.new(1024, 0_u8)
      @released = false

      def initialize(fixture : B::Fixture, cache : B::WeightCache)
        @resources = PixelUMMFFNAaProbe.resource_bytes
        @named_bytes = @resources.values.sum(0_i64)
        raise ArgumentError.new("FFN A/A named-memory cap exceeded") unless @named_bytes <= MAX_NAMED_BYTES
        records = fixture.windows["w0"].captures["block0"]
        raise "Metal unavailable" unless ML::Metal::Device.init!
        begin
          @resources.each { |name, bytes| @buffers[name] = ML::MetalBuffer.new(bytes) }
          {
            "input.norm" => "post_attn_norm", "expected.gate" => "gate_proj",
            "expected.up" => "up_proj", "expected.product" => "product",
            "expected.down" => "down_proj",
          }.each do |name, stage|
            fixture.tensor_file.upload_into!(records[stage], buffer(name))
          end
          {"gate", "up", "down"}.each do |stage|
            cache.tensor_file.upload_into!(cache.weights["block0.mlp.#{stage}_proj.weight"], buffer("weight.#{stage}"))
          end
          @buffers.each do |name, value|
            if name.starts_with?("work.") || name.starts_with?("capture.") || name == "canonical.down"
              Slice.new(value.contents.as(UInt8*), value.size.to_i32).fill(255_u8)
            elsif name == "metrics"
              Slice.new(value.contents.as(UInt8*), value.size.to_i32).fill(0_u8)
            end
          end
          @plan = FFNAaFFI.create(ML::Metal::Device.instance.device_handle,
            buffer("input.norm").handle, buffer("weight.gate").handle,
            buffer("weight.up").handle, buffer("weight.down").handle,
            buffer("work.gate").handle, buffer("work.up").handle,
            buffer("work.down").handle, buffer("capture.gate").handle,
            buffer("capture.up").handle, 5760, @error_text.to_unsafe, @error_text.size)
          raise "FFN A/A preparation: #{String.new(@error_text.to_unsafe)}" if @plan.null?
          @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16", SOURCE)
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def buffer(name : String) : ML::MetalBuffer
        @buffers[name]
      end

      def encode!(chain : ML::Metal::MPSCommandChain, capture_raw : Bool) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        status = FFNAaFFI.encode(@plan, chain.handle, capture_raw ? 1 : 0,
          @error_text.to_unsafe, @error_text.size)
        chain.resume!
        raise "FFN A/A encode: #{status}: #{String.new(@error_text.to_unsafe)}" unless status == 0
      end

      def copy_canonical!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
        raise "FFN A/A canonical blit unavailable" if blit.null?
        MetalDispatchFFI.blit_copy_buffer(blit, buffer("work.down").handle, 0_i64,
          buffer("canonical.down").handle, 0_i64, buffer("work.down").size)
        MetalDispatchFFI.blit_end_encoding(blit)
        chain.resume!
      end

      def compare_sample!(chain : ML::Metal::MPSCommandChain, sample : Int32) : Nil
        compare!(chain, "work.up", "expected.product", sample * 3, 5760 * 12288)
        compare!(chain, "work.down", "expected.down", sample * 3 + 1, 5760 * 4096)
        compare!(chain, "work.down", "canonical.down", sample * 3 + 2, 5760 * 4096)
      end

      def controls!(chain : ML::Metal::MPSCommandChain) : Nil
        compare!(chain, "capture.gate", "expected.gate", 54, 5760 * 12288)
        compare!(chain, "capture.up", "expected.up", 55, 5760 * 12288)
        compare!(chain, "canonical.down", "canonical.down", 56, 5760 * 4096, inject: true)
      end

      private def compare!(chain : ML::Metal::MPSCommandChain, actual : String,
                           expected : String, slot : Int32, count : Int32, inject = false) : Nil
        @lifecycle.mark_attempt!
        chain.encoder.set_pipeline(@pipeline.not_nil!)
          .set_buffer(buffer(actual), 0)
          .set_buffer(buffer(expected), 1)
          .set_buffer(buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite, slot.to_i64 * 16)
          .set_value(count.to_u32, 3).set_value(count.to_u32, 4)
          .set_value(inject ? 1_u32 : 0_u32, 5)
          .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(C::Metric)
        raise "FFN A/A readback requires sole terminal wait" unless chain.completed? && chain.wait_count == 1
        words = Array(UInt32).new(57 * 4, 0_u32)
        buffer("metrics").read_bytes(words.to_unsafe.as(UInt8*), words.size * sizeof(UInt32))
        Array.new(57) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def counts : Array(UInt64)
        values = Array(UInt64).new(7, 0_u64)
        raise "FFN A/A stats failed" unless FFNAaFFI.stats(@plan, values.to_unsafe, values.size) == 0
        values
      end

      def postwait!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_terminal_wait!(chain.completed?, chain.wait_count)
      end

      def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
        raise "FFN A/A unsafe release" unless @lifecycle.release_allowed?(chain.completed?, chain.wait_count)
        release_resources!
      end

      def release_unsubmitted! : Nil
        raise "FFN A/A cannot release submitted resources" unless @lifecycle.release_allowed?(false, 0)
        release_resources!
      end

      private def release_resources! : Nil
        return if @released
        FFNAaFFI.release(@plan) unless @plan.null?
        @plan = Pointer(Void).null
        @buffers.each_value(&.release)
        @buffers.clear
        @released = true
      end
    end
  {% end %}
end
