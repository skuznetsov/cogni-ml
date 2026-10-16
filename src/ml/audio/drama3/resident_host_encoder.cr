{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  require "./host_profile"

  module ML::Audio::Drama3::Resident
    # Fixed-cost host profiling wrapper for resident encoder API calls. The
    # underlying encoder remains responsible for every Metal operation.
    struct HostEncoder
      @encoder : ML::Metal::ComputeEncoder
      @profile : ML::Audio::Drama3::HostProfile
      @weight_probe : ML::Audio::Drama3::MetalBackend::WeightBufferBank?

      def initialize(command : ML::Metal::CommandBuffer, @profile : ML::Audio::Drama3::HostProfile,
                     @weight_probe : ML::Audio::Drama3::MetalBackend::WeightBufferBank? = nil)
        @encoder = @profile.measure(ML::Audio::Drama3::HostEvent::EncoderCreate) do
          ML::Metal::ComputeEncoder.new(command)
        end
      end

      def set_pipeline(pipeline : ML::Metal::ComputePipeline) : self
        @profile.add(ML::Audio::Drama3::HostCounter::PipelineBindings, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.set_pipeline(pipeline)
        end
        self
      end

      def set_buffer(buffer : ML::MetalBuffer, index : Int32,
                     access : ML::Metal::BufferAccess = ML::Metal::BufferAccess::Read,
                     offset : Int64 = 0, length : Int64 = -1, partition : Int32 = -1) : self
        @profile.add(ML::Audio::Drama3::HostCounter::BufferBindings, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.set_buffer(buffer, index, access, offset, length, partition)
        end
        @weight_probe.try(&.record_binding(buffer))
        self
      end

      def set_bytes(data : Pointer(Void), length : Int32, index : Int32) : self
        @profile.add(ML::Audio::Drama3::HostCounter::ByteBindings, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.set_bytes(data, length, index)
        end
        self
      end

      def dispatch_1d(count : Int32, threadgroup_size : Int32 = 256) : self
        @profile.add(ML::Audio::Drama3::HostCounter::KernelDispatches, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.dispatch_1d(count, threadgroup_size)
        end
        self
      end

      def dispatch_threadgroups(threadgroup_count : {Int32, Int32, Int32},
                                threadgroup_size : {Int32, Int32, Int32}) : self
        @profile.add(ML::Audio::Drama3::HostCounter::KernelDispatches, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.dispatch_threadgroups(threadgroup_count, threadgroup_size)
        end
        self
      end

      def set_threadgroup_memory(length : Int32, index : Int32) : self
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.set_threadgroup_memory(length, index)
        end
        self
      end

      def end_encoding : Nil
        @profile.add(ML::Audio::Drama3::HostCounter::EncoderEnds, 1_i64)
        @profile.measure(ML::Audio::Drama3::HostEvent::EncoderAPI) do
          @encoder.end_encoding
        end
      end
    end
  end
{% end %}
