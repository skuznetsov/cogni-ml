# Caller-chain Euler scheduler for the native first-segment Hy-Nomas path.
# Schedule values are built in setup and uploaded once; each forward writes
# the model timestep into the refiner's existing scalar, then updates the
# channels-first latent in place with the upstream BF16 boundary round-trip.
require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video
  class HyNomasEulerPlan
    MAX_STEPS = 8

    record Step, sigma : Float32, timestep : Float32, dt : Float32

    SOURCE = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_euler.metal") }}

    @@timestep_pipeline : ML::Metal::ComputePipeline?
    @@update_pipeline : ML::Metal::ComputePipeline?
    @@initial_round_pipeline : ML::Metal::ComputePipeline?

    @latent : ML::MetalBuffer
    @velocity : ML::MetalBuffer
    @timestep_buffer : ML::MetalBuffer
    @schedule_arena : Arena
    @schedule_buffer : ML::MetalBuffer
    @steps : Array(Step)
    @element_count : Int32
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @next_step = 0
    @awaiting_update : Int32? = nil
    @initial_latent_rounded = false
    @released = false
    @poisoned = false

    getter step_count : Int32
    getter element_count : Int32

    def self.schedule(steps : Int32 = MAX_STEPS, shift : Float32 = 5.0_f32) : Array(Step)
      unless (1..MAX_STEPS).includes?(steps)
        raise ArgumentError.new("Hy-Nomas Euler steps must be between 1 and #{MAX_STEPS}")
      end
      unless shift.finite? && shift > 0.0_f32
        raise ArgumentError.new("Hy-Nomas Euler shift must be finite and positive")
      end

      sigmas = Array(Float32).new(steps + 1) do |index|
        timestep_fraction = (steps - index).to_f32 / steps.to_f32
        if shift == 1.0_f32
          timestep_fraction
        else
          numerator = shift * timestep_fraction
          denominator = 1.0_f32 + (shift - 1.0_f32) * timestep_fraction
          numerator / denominator
        end
      end

      Array(Step).new(steps) do |index|
        sigma = sigmas[index]
        Step.new(sigma, sigma * 1000.0_f32, sigmas[index + 1] - sigma)
      end
    end

    def initialize(@latent : ML::MetalBuffer, @velocity : ML::MetalBuffer,
                   @timestep_buffer : ML::MetalBuffer, *,
                   steps : Int32 = MAX_STEPS, shift : Float32 = 5.0_f32)
      @steps = self.class.schedule(steps, shift)
      @step_count = @steps.size.to_i32
      unless @latent.valid? && @velocity.valid? && @timestep_buffer.valid? &&
             @latent.size > 0 && @latent.size % sizeof(Float32) == 0 &&
             @velocity.size == @latent.size &&
             @timestep_buffer.size == sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas Euler buffers must be exact F32 latent, velocity, and scalar timestep extents")
      end
      unless [@latent, @velocity, @timestep_buffer].all? do |buffer|
               buffer.storage_mode.shared?
             end
        raise ArgumentError.new("Hy-Nomas Euler buffers must use shared F32 storage")
      end
      count = @latent.size // sizeof(Float32)
      unless count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas Euler latent element count exceeds Int32")
      end
      @element_count = count.to_i32
      MetalOps.distinct(@latent, @velocity, @timestep_buffer)

      self.class.prepare!
      @schedule_arena = Arena.new
      schedule_values = Array(Float32).new(@step_count * 2)
      @steps.each do |step|
        schedule_values << step.timestep
        schedule_values << step.dt
      end
      @schedule_buffer = @schedule_arena.upload("hy_nomas.euler.schedule", schedule_values)
      MetalOps.distinct(@schedule_buffer, @latent, @velocity, @timestep_buffer)
    end

    def self.prepare! : Nil
      initial_round_pipeline
      timestep_pipeline
      update_pipeline
    end

    # Detached CPU oracle values; this allocation is setup/diagnostic-only.
    def schedule : Array(Step)
      ensure_live!
      @steps.dup
    end

    def timestep_at(step : Int32) : Float32
      @steps[validate_step!(step)].timestep
    end

    def dt_at(step : Int32) : Float32
      @steps[validate_step!(step)].dt
    end

    def resident_bytes : Int64
      ensure_live!
      @schedule_arena.resident_bytes
    end

    # Round the initial sample through BF16 once, then copy the indexed F32
    # timestep into the caller-owned scalar consumed by the text refiner. These
    # are GPU commands on the supplied encoder; they do not wait, read back, or
    # allocate.
    def encode_timestep(chain : ML::Metal::MPSCommandChain, step : Int32) : Nil
      validate_next!(chain, step, expect_update: false)
      begin
        unless @initial_latent_rounded
          chain.encoder.set_pipeline(self.class.initial_round_pipeline)
            .set_buffer(@latent, 0, ML::Metal::BufferAccess::ReadWrite)
            .set_value(@element_count.to_u32, 1)
            .dispatch_1d(@element_count)
          @initial_latent_rounded = true
        end
        chain.encoder.set_pipeline(self.class.timestep_pipeline)
          .set_buffer(@schedule_buffer, 0)
          .set_buffer(@timestep_buffer, 1, ML::Metal::BufferAccess::Write)
          .set_value(step.to_u32, 2)
          .dispatch_1d(1)
        @last_chain = chain
        @awaiting_update = step
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    # Round the F32 model output through BF16, perform the scheduler update in
    # F32, then round the updated latent through BF16 as upstream does.
    def encode_update(chain : ML::Metal::MPSCommandChain, step : Int32) : Nil
      validate_next!(chain, step, expect_update: true)
      begin
        chain.encoder.set_pipeline(self.class.update_pipeline)
          .set_buffer(@latent, 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(@velocity, 1)
          .set_buffer(@schedule_buffer, 2)
          .set_value(step.to_u32, 3)
          .set_value(@element_count.to_u32, 4)
          .dispatch_1d(@element_count)
        @awaiting_update = nil
        @next_step += 1
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete the Hy-Nomas Euler command chain before release") unless chain.completed?
      end
      @schedule_arena.release
      @released = true
    end

    private def validate_next!(chain : ML::Metal::MPSCommandChain, step : Int32,
                               *, expect_update : Bool) : Nil
      ensure_live!
      raise ArgumentError.new("Hy-Nomas Euler plan is poisoned") if @poisoned
      chain.encoder
      validate_step!(step)
      if previous = @last_chain
        unless previous.same?(chain)
          raise ArgumentError.new("Hy-Nomas Euler schedule must remain on one command chain")
        end
      end
      unless step == @next_step
        raise ArgumentError.new("Hy-Nomas Euler steps must be encoded in schedule order")
      end
      if expect_update
        unless @awaiting_update == step
          raise ArgumentError.new("Hy-Nomas Euler update requires its timestep to be encoded first")
        end
      elsif @awaiting_update
        raise ArgumentError.new("Hy-Nomas Euler timestep cannot advance before the prior update")
      end
      if @next_step != 0 || @awaiting_update
        # Validate live bindings before every append without reading their data.
        validate_bindings!
      end
    end

    private def validate_bindings! : Nil
      ensure_live!
      MetalOps.check(@latent, @element_count.to_i64)
      MetalOps.check(@velocity, @element_count.to_i64)
      MetalOps.check_storage(@timestep_buffer, 1_i64, WeightDType::F32)
      MetalOps.check_storage(@schedule_buffer, @step_count.to_i64 * 2,
        WeightDType::F32)
      unless @latent.size == @element_count.to_i64 * sizeof(Float32) &&
             @velocity.size == @latent.size &&
             @timestep_buffer.size == sizeof(Float32) &&
             @schedule_buffer.size == @step_count.to_i64 * 2 * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas Euler buffer extent changed")
      end
      MetalOps.distinct(@latent, @velocity, @timestep_buffer, @schedule_buffer)
    end

    private def validate_step!(step : Int32) : Int32
      unless (0...@step_count).includes?(step)
        raise ArgumentError.new("Hy-Nomas Euler step index is out of range")
      end
      step
    end

    private def ensure_live! : Nil
      raise ArgumentError.new("Hy-Nomas Euler plan has been released") if @released
    end

    def self.timestep_pipeline : ML::Metal::ComputePipeline
      @@timestep_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_euler_copy_timestep") do
        ML::Metal::ComputePipeline.new("hy_nomas_euler_copy_timestep", SOURCE)
      end
    end

    def self.initial_round_pipeline : ML::Metal::ComputePipeline
      @@initial_round_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_euler_round_initial_latent_bf16") do
        ML::Metal::ComputePipeline.new("hy_nomas_euler_round_initial_latent_bf16", SOURCE)
      end
    end

    def self.update_pipeline : ML::Metal::ComputePipeline
      @@update_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_euler_update_latent") do
        ML::Metal::ComputePipeline.new("hy_nomas_euler_update_latent", SOURCE)
      end
    end
  end
end
