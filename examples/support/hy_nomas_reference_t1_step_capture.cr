module ML::Video
  # Same-chain, allocation-free T1 checkpoint transform for the narrow
  # original-photo reference diagnostic. The outputs are BF16 values expanded
  # back into the F32 buffers used by the native timeline.
  class HyNomasReferenceT1StepCapturePlan
    ELEMENTS     = 32 * 30 * 9 * 16
    BUFFER_BYTES = ELEMENTS.to_i64 * sizeof(Float32)

    SOURCE = <<-METAL
      #include <metal_stdlib>
      using namespace metal;
      #pragma clang fp contract(off)

      static inline ushort hy_nomas_reference_capture_bf16(float value) {
        uint bits = as_type<uint>(value);
        uint magnitude = bits & 0x7fffffffu;
        if (magnitude > 0x7f800000u) {
          return ushort((bits >> 16u) | 0x0040u);
        }
        return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
      }

      static inline float hy_nomas_reference_capture_bf16_f32(float value) {
        return as_type<float>((uint)hy_nomas_reference_capture_bf16(value) << 16u);
      }

      kernel void hy_nomas_reference_t1_capture_velocity_x0_bf16(
          device const float *latent [[buffer(0)]],
          device const float *velocity [[buffer(1)]],
          device float *velocity_bf16 [[buffer(2)]],
          device float *x0_bf16 [[buffer(3)]],
          constant float &sigma [[buffer(4)]],
          constant uint &count [[buffer(5)]],
          uint tid [[thread_position_in_grid]]) {
        if (tid >= count) return;
        const float latent_bf16 = hy_nomas_reference_capture_bf16_f32(latent[tid]);
        const float velocity_effective = hy_nomas_reference_capture_bf16_f32(velocity[tid]);
        const float scaled_velocity = sigma * velocity_effective;
        velocity_bf16[tid] = velocity_effective;
        x0_bf16[tid] = hy_nomas_reference_capture_bf16_f32(latent_bf16 - scaled_velocity);
      }
    METAL

    @element_count : Int32
    @pipeline : ML::Metal::ComputePipeline

    def self.bf16_expanded_f32(value : Float32) : Float32
      bits = value.unsafe_as(UInt32)
      magnitude = bits & 0x7fff_ffff_u32
      rounded = if magnitude > 0x7f80_0000_u32
                  (bits >> 16) | 0x0040_u32
                else
                  (bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)) >> 16
                end
      (rounded << 16).unsafe_as(Float32)
    end

    def self.x0_hat_bf16_expanded_f32(latent : Float32, velocity : Float32,
                                      sigma : Float32) : Float32
      latent_bf16 = bf16_expanded_f32(latent)
      velocity_bf16 = bf16_expanded_f32(velocity)
      bf16_expanded_f32(latent_bf16 - sigma * velocity_bf16)
    end

    def initialize(@element_count : Int32 = ELEMENTS)
      unless @element_count == ELEMENTS
        raise ArgumentError.new("reference T1 step capture requires the fixed [1, 32, 30, 9, 16] geometry")
      end
      @pipeline = ML::Metal::PipelineCache.get("hy_nomas_reference_t1_capture_velocity_x0_bf16") do
        ML::Metal::ComputePipeline.new("hy_nomas_reference_t1_capture_velocity_x0_bf16", SOURCE)
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder,
               latent : ML::MetalBuffer, velocity : ML::MetalBuffer,
               velocity_bf16 : ML::MetalBuffer, x0_bf16 : ML::MetalBuffer,
               sigma : Float32) : Nil
      unless sigma.finite? && sigma >= 0_f32 && sigma <= 1_f32
        raise ArgumentError.new("reference T1 step capture requires a finite schedule sigma")
      end
      buffers = {latent, velocity, velocity_bf16, x0_bf16}
      unless buffers.all? do |buffer|
               buffer.valid? && buffer.size == BUFFER_BYTES && buffer.storage_mode.shared?
             end
        raise ArgumentError.new("reference T1 step capture requires exact shared F32 buffers")
      end
      MetalOps.distinct(latent, velocity, velocity_bf16, x0_bf16)
      encoder.set_pipeline(@pipeline)
        .set_buffer(latent, 0)
        .set_buffer(velocity, 1)
        .set_buffer(velocity_bf16, 2, ML::Metal::BufferAccess::Write)
        .set_buffer(x0_bf16, 3, ML::Metal::BufferAccess::Write)
        .set_value(sigma, 4)
        .set_value(@element_count.to_u32, 5)
        .dispatch_1d(@element_count)
    end
  end
end
