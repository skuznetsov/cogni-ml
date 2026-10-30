require "../core/buffer"
require "../metal/device"
require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::HyNomasTimelineOps
  GATHER_KERNEL      = "hy_nomas_timeline_gather_f32"
  SCATTER_KERNEL     = "hy_nomas_timeline_scatter_f32"
  COPY_KERNEL        = "hy_nomas_timeline_copy_f32"
  CACHE_NOISE_KERNEL = "hy_nomas_timeline_cache_noise_bf16_f32"
  SOURCE             = {{ read_file("#{__DIR__}/kernels/hy_nomas_timeline_ops.metal") }}

  # Full resident banks describe [1,32,180,9,16], six 30-latent segments,
  # and six global-offset [4320,128] RoPE tables. A run may process a prefix
  # of two through six segments without shrinking or resampling those banks.
  # Source: hy-nomas-20261024 @ 911cea6458a89c2663513ff7099561ec190dd491.
  # The cache-noise constants are pinned to the distilled four-step
  # configs/experiment/inference.yaml:91-92 (cache_at_step=-1,
  # cache_noise_sigma=0.206897); this path is not the t=0 cache contract.
  class Geometry
    DEFAULT_TOTAL_LATENT_FRAMES   =          180
    DEFAULT_SEGMENT_LATENT_FRAMES =           30
    DEFAULT_CHANNELS              =           32
    DEFAULT_LATENT_HEIGHT         =            9
    DEFAULT_LATENT_WIDTH          =           16
    DEFAULT_ROPE_HEAD_DIM         =          128
    CACHE_SIGMA                   = 0.206897_f32
    CACHE_TIMESTEP                =  206.897_f32

    getter total_latent_frames : Int32
    getter segment_latent_frames : Int32
    getter channels : Int32
    getter latent_height : Int32
    getter latent_width : Int32
    getter rope_head_dim : Int32

    def initialize(*,
                   @total_latent_frames : Int32 = DEFAULT_TOTAL_LATENT_FRAMES,
                   @segment_latent_frames : Int32 = DEFAULT_SEGMENT_LATENT_FRAMES,
                   @channels : Int32 = DEFAULT_CHANNELS,
                   @latent_height : Int32 = DEFAULT_LATENT_HEIGHT,
                   @latent_width : Int32 = DEFAULT_LATENT_WIDTH,
                   @rope_head_dim : Int32 = DEFAULT_ROPE_HEAD_DIM)
      validate!
    end

    def segment_count : Int32
      @total_latent_frames // @segment_latent_frames
    end

    def elements_per_latent_frame : Int32
      checked_product(@channels, @latent_height, @latent_width).to_i32
    end

    def segment_elements : Int32
      checked_product(@segment_latent_frames, elements_per_latent_frame).to_i32
    end

    def timeline_elements : Int32
      checked_product(@total_latent_frames, elements_per_latent_frame).to_i32
    end

    def rope_tokens_per_segment : Int32
      checked_product(@segment_latent_frames, @latent_height, @latent_width).to_i32
    end

    def rope_elements_per_segment : Int32
      checked_product(rope_tokens_per_segment, @rope_head_dim).to_i32
    end

    def rope_bank_elements : Int32
      checked_product(segment_count, rope_elements_per_segment).to_i32
    end

    def segment_start(segment_index : Int32) : Int32
      checked_segment_index(segment_index)
      (segment_index.to_i64 * @segment_latent_frames).to_i32
    end

    def rope_offset(segment_index : Int32) : Int32
      checked_segment_index(segment_index)
      (segment_index.to_i64 * rope_elements_per_segment).to_i32
    end

    def cache_noise_offset(segment_index : Int32) : Int32
      checked_segment_index(segment_index)
      (segment_index.to_i64 * segment_elements).to_i32
    end

    def timeline_bytes : Int64
      timeline_elements.to_i64 * sizeof(Float32)
    end

    def segment_bytes : Int64
      segment_elements.to_i64 * sizeof(Float32)
    end

    def cache_noise_bank_bytes : Int64
      timeline_elements.to_i64 * sizeof(UInt16)
    end

    def rope_bank_bytes : Int64
      rope_bank_elements.to_i64 * sizeof(Float32)
    end

    def cache_sigma : Float32
      CACHE_SIGMA
    end

    def cache_timestep : Float32
      CACHE_TIMESTEP
    end

    private def validate! : Nil
      unless @total_latent_frames > 0 && @segment_latent_frames > 0 &&
             @channels > 0 && @latent_height > 0 && @latent_width > 0 && @rope_head_dim > 0
        raise ArgumentError.new("Hy-Nomas timeline dimensions must be positive")
      end
      unless @total_latent_frames % @segment_latent_frames == 0
        raise ArgumentError.new("Hy-Nomas segment length must divide the resident latent timeline")
      end
      unless CACHE_SIGMA > 0.0_f32 && CACHE_SIGMA <= 1.0_f32 &&
             CACHE_TIMESTEP == CACHE_SIGMA * 1000.0_f32
        raise ArgumentError.new("Hy-Nomas cache-noise timestep does not match the pinned sigma")
      end
      # These bounds cover the metal dispatch counts and all uint kernel
      # parameters before a plan can bind any buffers.
      counts = [timeline_elements, segment_elements, rope_bank_elements]
      unless counts.all? { |count| count > 0 && count <= Int32::MAX }
        raise ArgumentError.new("Hy-Nomas timeline geometry exceeds the Metal dispatch limit")
      end
      [@total_latent_frames, @segment_latent_frames, @channels, @latent_height,
       @latent_width, @rope_head_dim, rope_tokens_per_segment, rope_elements_per_segment]
        .each do |value|
          unless value > 0 && value.to_i64 <= UInt32::MAX
            raise ArgumentError.new("Hy-Nomas timeline parameter exceeds the Metal uint limit")
          end
        end
    end

    private def checked_segment_index(index : Int32) : Nil
      unless 0 <= index < segment_count
        raise ArgumentError.new("Hy-Nomas segment index is outside the resident timeline")
      end
    end

    private def checked_product(*values : Int32) : Int64
      product = 1_i64
      values.each do |value|
        raise ArgumentError.new("Hy-Nomas timeline dimensions must be positive") unless value > 0
        if product > Int32::MAX.to_i64 // value
          raise ArgumentError.new("Hy-Nomas timeline geometry exceeds the Metal dispatch limit")
        end
        product *= value
      end
      product
    end
  end

  # Shape contract for copying temporal prefix frames from channels-first
  # [1,C,T,H,W] storage into a compact [1,C,prefix,H,W] destination.
  class PrefixGeometry
    getter channels : Int32
    getter total_frames : Int32
    getter prefix_frames : Int32
    getter height : Int32
    getter width : Int32

    def initialize(*,
                   @channels : Int32,
                   @total_frames : Int32,
                   @prefix_frames : Int32,
                   @height : Int32,
                   @width : Int32)
      validate!
    end

    def elements_per_frame : Int32
      checked_product(@channels, @height, @width).to_i32
    end

    def source_elements : Int32
      checked_product(@total_frames, elements_per_frame).to_i32
    end

    def prefix_elements : Int32
      checked_product(@prefix_frames, elements_per_frame).to_i32
    end

    def source_bytes : Int64
      source_elements.to_i64 * sizeof(Float32)
    end

    def prefix_bytes : Int64
      prefix_elements.to_i64 * sizeof(Float32)
    end

    private def validate! : Nil
      unless @channels > 0 && @total_frames > 0 && @prefix_frames > 0 &&
             @height > 0 && @width > 0
        raise ArgumentError.new("Hy-Nomas prefix dimensions must be positive")
      end
      unless @prefix_frames <= @total_frames
        raise ArgumentError.new("Hy-Nomas prefix frame count cannot exceed source frames")
      end
      counts = [elements_per_frame, source_elements, prefix_elements]
      unless counts.all? { |count| count > 0 && count <= Int32::MAX }
        raise ArgumentError.new("Hy-Nomas prefix geometry exceeds the Metal dispatch limit")
      end
    end

    private def checked_product(*values : Int32) : Int64
      product = 1_i64
      values.each do |value|
        raise ArgumentError.new("Hy-Nomas prefix dimensions must be positive") unless value > 0
        if product > Int32::MAX.to_i64 // value
          raise ArgumentError.new("Hy-Nomas prefix geometry exceeds the Metal dispatch limit")
        end
        product *= value
      end
      product
    end
  end

  # Reusable prefix gather over the same preallocated GPU buffers and serial
  # chain as generation. It intentionally owns no tensor allocations.
  class PrefixPlan
    @source : ML::MetalBuffer?
    @destination : ML::MetalBuffer?
    @pipeline : ML::Metal::ComputePipeline
    @chain_object_id : UInt64?
    @encoded = false
    @released = false

    getter geometry : PrefixGeometry

    def initialize(@geometry : PrefixGeometry, *, source : ML::MetalBuffer,
                   destination : ML::MetalBuffer)
      @source = source
      @destination = destination
      validate_buffers!
      @pipeline = ML::Metal::PipelineCache.get(GATHER_KERNEL) do
        ML::Metal::ComputePipeline.new(GATHER_KERNEL, SOURCE)
      end
    end

    def plan_buffer_bytes : Int64
      0_i64
    end

    def encoded? : Bool
      @encoded
    end

    def released? : Bool
      @released
    end

    # Encode a prefix gather at global frame zero. No allocation, wait, or
    # readback occurs here; caller retains this plan through terminal drain.
    def encode!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas prefix plan is already released") if @released
      raise ArgumentError.new("Hy-Nomas prefix plan can be encoded only once") if @encoded
      bind_live_chain!(chain)
      source = live_source!
      destination = live_destination!
      # Once bound to the live chain, the plan must remain releasable even if
      # a later encoder call raises after partial command encoding.
      @encoded = true
      chain.encoder.set_pipeline(@pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(0_u32, 2)
        .set_value(@geometry.total_frames.to_u32, 3)
        .set_value(@geometry.prefix_frames.to_u32, 4)
        .set_value(@geometry.height.to_u32, 5)
        .set_value(@geometry.width.to_u32, 6)
        .set_value(@geometry.prefix_elements.to_u32, 7)
        .dispatch_1d(@geometry.prefix_elements)
    end

    # Release references only after the shared chain's sole terminal wait.
    def release_after_drain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas prefix plan is already released") if @released
      bind_chain!(chain)
      unless @encoded && chain.drained? && chain.wait_count == 1
        raise ArgumentError.new("Hy-Nomas prefix release requires an encoded plan and exactly one terminal chain wait")
      end
      @source = nil
      @destination = nil
      @released = true
    end

    private def validate_buffers! : Nil
      source = @source || raise ArgumentError.new("Hy-Nomas prefix source is unavailable")
      destination = @destination || raise ArgumentError.new("Hy-Nomas prefix destination is unavailable")
      validate_f32!(source, @geometry.source_elements, @geometry.source_bytes, "source")
      validate_f32!(destination, @geometry.prefix_elements, @geometry.prefix_bytes, "destination")
      if MetalOps.overlap?(source, destination)
        raise ArgumentError.new("Hy-Nomas prefix source and destination must not overlap")
      end
    end

    private def validate_f32!(buffer : ML::MetalBuffer, elements : Int32,
                              bytes : Int64, name : String) : Nil
      MetalOps.check_storage(buffer, elements.to_i64, WeightDType::F32)
      unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared && buffer.size == bytes
        raise ArgumentError.new("Hy-Nomas prefix #{name} buffer must be exact-size shared F32 storage")
      end
    end

    private def live_source! : ML::MetalBuffer
      buffer = @source || raise ArgumentError.new("Hy-Nomas prefix source is unavailable")
      raise ArgumentError.new("Hy-Nomas prefix source buffer was released before chain drain") unless buffer.valid?
      buffer
    end

    private def live_destination! : ML::MetalBuffer
      buffer = @destination || raise ArgumentError.new("Hy-Nomas prefix destination is unavailable")
      raise ArgumentError.new("Hy-Nomas prefix destination buffer was released before chain drain") unless buffer.valid?
      buffer
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas prefix plan is already released") if @released
      identity = chain.object_id
      if prior = @chain_object_id
        unless prior == identity
          raise ArgumentError.new("Hy-Nomas prefix plan must use one serial command chain")
        end
      else
        @chain_object_id = identity
      end
    end

    private def bind_live_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      bind_chain!(chain)
      raise ArgumentError.new("Hy-Nomas prefix plan cannot enqueue after chain drain") if chain.drained?
    end
  end

  enum SchedulePhase
    Ready
    Preparing
    Prepared
    Denoised
    Scattered
    CacheNoiseWritten
    CacheForwardEnqueued
    Finished
    Failed
  end

  # CPU-only schedule contract. It models enqueue order, not GPU completion;
  # callers still retain all buffers and the plan until the chain's terminal
  # wait. The last active segment is scattered cleanly and skips cache write.
  class Schedule
    getter next_segment : Int32 = 0
    getter phase : SchedulePhase = SchedulePhase::Ready
    getter run_segments : Int32
    getter geometry : Geometry

    def initialize(@geometry : Geometry, *, @run_segments : Int32 = @geometry.segment_count)
      unless 2 <= @run_segments <= @geometry.segment_count
        raise ArgumentError.new("Hy-Nomas run segment count must be between two and the resident segment count")
      end
    end

    def cache_write_required?(segment_index : Int32) : Bool
      check_active_segment!(segment_index)
      segment_index < @run_segments - 1
    end

    def expect_prepare!(segment_index : Int32) : Nil
      check_not_failed_or_finished!
      unless @phase.ready? && segment_index == @next_segment && segment_index < @run_segments
        raise ArgumentError.new("Hy-Nomas prepare must target the next active segment in Ready phase")
      end
    end

    def begin_prepare!(segment_index : Int32) : Nil
      expect_prepare!(segment_index)
      @phase = SchedulePhase::Preparing
    end

    def record_prepared!(segment_index : Int32) : Nil
      expect_phase!(segment_index, SchedulePhase::Preparing)
      @phase = SchedulePhase::Prepared
    end

    def record_denoise_complete!(segment_index : Int32) : Nil
      expect_phase!(segment_index, SchedulePhase::Prepared)
      @phase = SchedulePhase::Denoised
    end

    def record_scattered!(segment_index : Int32) : Nil
      expect_phase!(segment_index, SchedulePhase::Denoised)
      @phase = SchedulePhase::Scattered
    end

    def expect_scatter!(segment_index : Int32) : Nil
      expect_phase!(segment_index, SchedulePhase::Denoised)
    end

    def expect_cache_noise_write!(segment_index : Int32) : Nil
      unless cache_write_required?(segment_index)
        raise ArgumentError.new("Final active segment skips the cache-noise write")
      end
      expect_phase!(segment_index, SchedulePhase::Scattered)
    end

    def record_cache_noise_written!(segment_index : Int32) : Nil
      expect_cache_noise_write!(segment_index)
      @phase = SchedulePhase::CacheNoiseWritten
    end

    def record_cache_forward_enqueued!(segment_index : Int32) : Nil
      unless cache_write_required?(segment_index)
        raise ArgumentError.new("Final active segment skips the cache forward")
      end
      expect_phase!(segment_index, SchedulePhase::CacheNoiseWritten)
      @phase = SchedulePhase::CacheForwardEnqueued
    end

    def advance_segment!(segment_index : Int32) : Nil
      check_not_failed_or_finished!
      unless segment_index == @next_segment && segment_index < @run_segments
        raise ArgumentError.new("Hy-Nomas advance must target the next active segment")
      end

      if cache_write_required?(segment_index)
        unless @phase.cache_forward_enqueued?
          message = @phase.scattered? ? "cache-noise write and cache forward are required before advance" : "cache forward is required before advance"
          raise ArgumentError.new("Hy-Nomas #{message}")
        end
      else
        unless @phase.scattered?
          raise ArgumentError.new("Hy-Nomas final active segment must be scattered before advance")
        end
      end

      @next_segment += 1
      @phase = @next_segment == @run_segments ? SchedulePhase::Finished : SchedulePhase::Ready
    end

    def fail! : Nil
      @phase = SchedulePhase::Failed
    end

    private def expect_phase!(segment_index : Int32, expected : SchedulePhase) : Nil
      check_not_failed_or_finished!
      unless segment_index == @next_segment && segment_index < @run_segments && @phase == expected
        raise ArgumentError.new("Hy-Nomas transition is invalid for the current segment or phase")
      end
    end

    private def check_active_segment!(segment_index : Int32) : Nil
      unless 0 <= segment_index < @run_segments
        raise ArgumentError.new("Hy-Nomas segment index is outside the active run")
      end
    end

    private def check_not_failed_or_finished! : Nil
      raise ArgumentError.new("Hy-Nomas schedule has failed") if @phase.failed?
      raise ArgumentError.new("Hy-Nomas schedule is finished") if @phase.finished?
    end
  end

  # Buffer holder for caller-owned, preallocated inputs/outputs. No Metal
  # buffers or constants are allocated by this plan; scalar constants are
  # passed with set_value during each dispatch.
  private class BufferSet
    getter timeline : ML::MetalBuffer
    getter segment_latent : ML::MetalBuffer
    getter cosine_bank : ML::MetalBuffer
    getter sine_bank : ML::MetalBuffer
    getter cosine : ML::MetalBuffer
    getter sine : ML::MetalBuffer
    getter cache_eps_bf16 : ML::MetalBuffer
    getter cache_timestep : ML::MetalBuffer

    def initialize(@timeline : ML::MetalBuffer, @segment_latent : ML::MetalBuffer,
                   @cosine_bank : ML::MetalBuffer, @sine_bank : ML::MetalBuffer,
                   @cosine : ML::MetalBuffer, @sine : ML::MetalBuffer,
                   @cache_eps_bf16 : ML::MetalBuffer, @cache_timestep : ML::MetalBuffer)
    end

    def all_valid? : Bool
      @timeline.valid? && @segment_latent.valid? && @cosine_bank.valid? &&
        @sine_bank.valid? && @cosine.valid? && @sine.valid? &&
        @cache_eps_bf16.valid? && @cache_timestep.valid?
    end
  end

  class Plan
    @buffers : BufferSet?
    @schedule : Schedule
    @run_segments : Int32
    @chain_object_id : UInt64?
    @released = false
    @gather_pipeline : ML::Metal::ComputePipeline
    @scatter_pipeline : ML::Metal::ComputePipeline
    @copy_pipeline : ML::Metal::ComputePipeline
    @cache_noise_pipeline : ML::Metal::ComputePipeline

    getter geometry : Geometry
    getter schedule : Schedule
    getter run_segments : Int32

    def initialize(@geometry : Geometry,
                   *, timeline : ML::MetalBuffer, segment_latent : ML::MetalBuffer,
                   cosine_bank : ML::MetalBuffer, sine_bank : ML::MetalBuffer,
                   cosine : ML::MetalBuffer, sine : ML::MetalBuffer,
                   cache_eps_bf16 : ML::MetalBuffer,
                   cache_timestep : ML::MetalBuffer,
                   @run_segments : Int32 = @geometry.segment_count)
      @schedule = Schedule.new(@geometry, run_segments: @run_segments)
      @buffers = BufferSet.new(timeline, segment_latent, cosine_bank, sine_bank,
        cosine, sine, cache_eps_bf16, cache_timestep)
      validate_buffers!
      @gather_pipeline = ML::Metal::PipelineCache.get(GATHER_KERNEL) do
        ML::Metal::ComputePipeline.new(GATHER_KERNEL, SOURCE)
      end
      @scatter_pipeline = ML::Metal::PipelineCache.get(SCATTER_KERNEL) do
        ML::Metal::ComputePipeline.new(SCATTER_KERNEL, SOURCE)
      end
      @copy_pipeline = ML::Metal::PipelineCache.get(COPY_KERNEL) do
        ML::Metal::ComputePipeline.new(COPY_KERNEL, SOURCE)
      end
      @cache_noise_pipeline = ML::Metal::PipelineCache.get(CACHE_NOISE_KERNEL) do
        ML::Metal::ComputePipeline.new(CACHE_NOISE_KERNEL, SOURCE)
      end
    end

    # Plan-owned resident buffer bytes; all tensor buffers belong to the caller.
    def plan_buffer_bytes : Int64
      0_i64
    end

    # Gather the current segment's channel-first latent and copy its global
    # RoPE cosine/sine tables into fixed 30-frame buffers on the same chain.
    def prepare_segment!(chain : ML::Metal::MPSCommandChain, segment_index : Int32) : Nil
      @schedule.expect_prepare!(segment_index)
      bind_live_chain!(chain)
      buffers = live_buffers!
      e = chain.encoder
      @schedule.begin_prepare!(segment_index)
      begin
        e.set_pipeline(@gather_pipeline)
          .set_buffer(buffers.timeline, 0)
          .set_buffer(buffers.segment_latent, 1, ML::Metal::BufferAccess::Write)
          .set_value(@geometry.segment_start(segment_index).to_u32, 2)
          .set_value(@geometry.total_latent_frames.to_u32, 3)
          .set_value(@geometry.segment_latent_frames.to_u32, 4)
          .set_value(@geometry.latent_height.to_u32, 5)
          .set_value(@geometry.latent_width.to_u32, 6)
          .set_value(@geometry.segment_elements.to_u32, 7)
          .dispatch_1d(@geometry.segment_elements)
        encode_rope_slice(e, buffers.cosine_bank, buffers.cosine, segment_index)
        encode_rope_slice(e, buffers.sine_bank, buffers.sine, segment_index)
      rescue ex
        @schedule.fail!
        raise ex
      end
      @schedule.record_prepared!(segment_index)
    end

    # The parent queues this only after the denoise steps for the active
    # segment are in the chain. The transition itself performs no GPU work.
    def mark_denoise_complete!(chain : ML::Metal::MPSCommandChain, segment_index : Int32) : Nil
      bind_live_chain!(chain)
      @schedule.record_denoise_complete!(segment_index)
    end

    # Scatter the clean generated segment before reusing its fixed buffer for
    # the cache-only noisy input. This keeps final exported latents unchanged.
    def scatter_generated!(chain : ML::Metal::MPSCommandChain, segment_index : Int32) : Nil
      @schedule.expect_scatter!(segment_index)
      bind_live_chain!(chain)
      buffers = live_buffers!
      e = chain.encoder
      begin
        e.set_pipeline(@scatter_pipeline)
          .set_buffer(buffers.segment_latent, 0)
          .set_buffer(buffers.timeline, 1, ML::Metal::BufferAccess::ReadWrite)
          .set_value(@geometry.segment_start(segment_index).to_u32, 2)
          .set_value(@geometry.total_latent_frames.to_u32, 3)
          .set_value(@geometry.segment_latent_frames.to_u32, 4)
          .set_value(@geometry.latent_height.to_u32, 5)
          .set_value(@geometry.latent_width.to_u32, 6)
          .set_value(@geometry.segment_elements.to_u32, 7)
          .dispatch_1d(@geometry.segment_elements)
      rescue ex
        @schedule.fail!
        raise ex
      end
      @schedule.record_scattered!(segment_index)
    end

    # Apply upstream's separate F32 multiply/add operations followed by BF16
    # output rounding in-place. The bank is raw BF16 and is exactly sized in
    # setup validation. The final active segment must skip this cache-only path.
    def write_cache_noise!(chain : ML::Metal::MPSCommandChain, segment_index : Int32) : Nil
      @schedule.expect_cache_noise_write!(segment_index)
      bind_live_chain!(chain)
      buffers = live_buffers!
      e = chain.encoder
      begin
        e.set_pipeline(@cache_noise_pipeline)
          .set_buffer(buffers.segment_latent, 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(buffers.cache_eps_bf16, 1)
          .set_buffer(buffers.cache_timestep, 2, ML::Metal::BufferAccess::Write)
          .set_value(segment_index.to_u32, 3)
          .set_value(@geometry.segment_elements.to_u32, 4)
          .set_value(@geometry.cache_sigma, 5)
          .set_value(@geometry.cache_timestep, 6)
          .dispatch_1d(@geometry.segment_elements)
      rescue ex
        @schedule.fail!
        raise ex
      end
      @schedule.record_cache_noise_written!(segment_index)
    end

    # Call after the parent has queued the native update_cache forward on this
    # same serial chain. This records enqueue order only; it does not wait.
    def record_cache_forward_enqueued!(chain : ML::Metal::MPSCommandChain,
                                       segment_index : Int32) : Nil
      bind_live_chain!(chain)
      @schedule.record_cache_forward_enqueued!(segment_index)
    end

    def advance_segment!(chain : ML::Metal::MPSCommandChain, segment_index : Int32) : Nil
      bind_live_chain!(chain)
      @schedule.advance_segment!(segment_index)
    end

    # Drop references to caller-owned buffers only after all active segments
    # have drained. It deliberately permits cleanup after an interrupted run:
    # once the terminal wait returns, even a partially scheduled plan no
    # longer owns in-flight GPU reads or writes.
    def release_after_drain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas plan is already released") if @released
      bind_chain!(chain)
      unless chain.drained? && chain.wait_count == 1
        raise ArgumentError.new("Hy-Nomas plan release requires exactly one terminal chain wait")
      end
      @buffers = nil
      @released = true
    end

    private def encode_rope_slice(e : ML::Metal::ComputeEncoder,
                                  source : ML::MetalBuffer, destination : ML::MetalBuffer,
                                  segment_index : Int32) : Nil
      e.set_pipeline(@copy_pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(@geometry.rope_offset(segment_index).to_u32, 2)
        .set_value(@geometry.rope_elements_per_segment.to_u32, 3)
        .dispatch_1d(@geometry.rope_elements_per_segment)
    end

    private def validate_buffers! : Nil
      buffers = @buffers || raise ArgumentError.new("Hy-Nomas timeline buffers are unavailable")
      validate_f32!(buffers.timeline, @geometry.timeline_elements, "timeline")
      validate_f32!(buffers.segment_latent, @geometry.segment_elements, "segment latent")
      validate_f32!(buffers.cosine_bank, @geometry.rope_bank_elements, "cosine bank")
      validate_f32!(buffers.sine_bank, @geometry.rope_bank_elements, "sine bank")
      validate_f32!(buffers.cosine, @geometry.rope_elements_per_segment, "cosine slice")
      validate_f32!(buffers.sine, @geometry.rope_elements_per_segment, "sine slice")
      MetalOps.check_storage(buffers.cache_eps_bf16, @geometry.timeline_elements.to_i64, WeightDType::BF16)
      validate_exact_storage!(buffers.cache_eps_bf16,
        @geometry.timeline_elements.to_i64 * sizeof(UInt16), "BF16 cache-noise bank")
      MetalOps.check_storage(buffers.cache_timestep, 1_i64, WeightDType::F32)
      validate_exact_storage!(buffers.cache_timestep, sizeof(Float32).to_i64, "cache timestep")

      buffers_list = [buffers.timeline, buffers.segment_latent, buffers.cosine_bank,
                      buffers.sine_bank, buffers.cosine, buffers.sine,
                      buffers.cache_eps_bf16, buffers.cache_timestep]
      buffers_list.each_with_index do |left, left_index|
        buffers_list.each_with_index do |right, right_index|
          next if right_index <= left_index
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas timeline plan buffers must be pairwise distinct")
          end
        end
      end
    end

    private def validate_f32!(buffer : ML::MetalBuffer, elements : Int32, name : String) : Nil
      MetalOps.check_storage(buffer, elements.to_i64, WeightDType::F32)
      validate_exact_storage!(buffer, elements.to_i64 * sizeof(Float32), name)
    end

    private def validate_exact_storage!(buffer : ML::MetalBuffer, bytes : Int64, name : String) : Nil
      unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared && buffer.size == bytes
        raise ArgumentError.new("Hy-Nomas #{name} buffer must be exact-size shared storage")
      end
    end

    private def live_buffers! : BufferSet
      raise ArgumentError.new("Hy-Nomas plan is released") if @released
      buffers = @buffers || raise ArgumentError.new("Hy-Nomas timeline buffers are unavailable")
      unless buffers.all_valid?
        raise ArgumentError.new("Hy-Nomas timeline plan buffer was released before chain drain")
      end
      buffers
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas plan is released") if @released
      identity = chain.object_id
      if prior = @chain_object_id
        unless prior == identity
          raise ArgumentError.new("Hy-Nomas timeline plan must use one serial command chain")
        end
      else
        @chain_object_id = identity
      end
    end

    private def bind_live_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      bind_chain!(chain)
      raise ArgumentError.new("Hy-Nomas timeline plan cannot enqueue after chain drain") if chain.drained?
    end
  end
end
