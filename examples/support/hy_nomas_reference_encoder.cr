require "./hy_nomas_probe_instrumentation"
require "./hy_nomas_reference_image"
require "../../src/ml/video/bf16_weight_cache"
require "../../src/ml/video/dcae_v_encoder"

module HyNomasReferenceEncoder
  # Converted encoder safetensors admitted for this experimental T0 path.
  ENCODER_SAFETENSORS_SHA256 = "c77e705b3d1534f1d3fcca68d146c9eecd479be5d01110aadee7b81ddc05ed50"
  # The recovered container independently matched all 169 pinned BF16 tensor
  # payloads; preserve the original container identity as a separate admission.
  RECOVERED_ENCODER_SAFETENSORS_SHA256 = "d0bbacbebe8960b147388d0bf61ec8bbc0eb8d575574d6ec5c5abe4878c8d6c1"
  FIXED_CFG                            = ML::Video::DCAEVEncoderConfig.new(288, 512, 120)
  MPS_MEMORY_SCOPE                     = "MPSGraph private workspace is opaque to MetalBuffer accounting; excluded from named-byte totals and not assumed zero"
  DIT_ENCODE_METRIC_KEYS               = %w(dit_encode_allocations dit_encode_readbacks dit_preterminal_waits)

  # Counts named MetalBuffer residency only. The native MPSGraph plan set is
  # created separately and its private workspace is opaque, not a zero-byte
  # item in this total.
  record ResidencyPlan, weight_bytes : Int64, workspace_bytes : Int64,
    clip_bytes : Int64, rgb_bytes : Int64, latent_bytes : Int64,
    named_bytes : Int64 do
    def self.for(cfg : ML::Video::DCAEVEncoderConfig) : self
      unless cfg.batches == 1 && cfg.width == 512 && cfg.height == 288 && cfg.frames == 120
        raise ArgumentError.new("reference encoder only admits the fixed 512x288x120 configuration")
      end
      weights = ML::Video::DCAEVEncoder.weight_bytes(cfg)
      workspace = ML::Video::DCAEVEncoder.workspace_required_bytes(cfg)
      clip = cfg.input_bytes
      rgb = cfg.rgb_bytes
      latent = cfg.output_bytes
      total = checked_add("encoder workspace and weights", weights, workspace)
      total = checked_add("encoder clip residency", total, clip)
      total = checked_add("encoder RGB residency", total, rgb)
      total = checked_add("encoder latent residency", total, latent)
      new(weights, workspace, clip, rgb, latent, total)
    end

    private def self.checked_add(label : String, left : Int64, right : Int64) : Int64
      unless left >= 0 && right >= 0 && right <= Int64::MAX - left
        raise ArgumentError.new("#{label} byte count overflows")
      end
      left + right
    end
  end

  # The reference occupies technical segment T0; generated segments begin at
  # index 1. Generated nonfinal segments get a cache writer, while the final
  # generated segment has no successor that needs its cache.
  record ReferenceHistoryPlan, total_segments : Int32 do
    def self.for(total_segments : Int32) : self
      unless 2 <= total_segments <= 6
        raise ArgumentError.new("reference history requires two through six total segments")
      end
      new(total_segments)
    end

    def generated_segment_indices : Array(Int32)
      (1...@total_segments).to_a
    end

    def generated_cache_writer_segment_indices : Array(Int32)
      @total_segments <= 2 ? [] of Int32 : (1...(@total_segments - 1)).to_a
    end

    def final_generated_segment_index : Int32
      @total_segments - 1
    end

    def generated_duration_seconds : Int32
      generated_segment_indices.size * 5
    end

    def total_timeline_duration_seconds : Int32
      @total_segments * 5
    end
  end

  def self.validate_budget!(plan : ResidencyPlan, budget_bytes : Int64) : Nil
    unless budget_bytes > 0 && plan.named_bytes <= budget_bytes
      raise ArgumentError.new("reference encoder peak named buffers exceed budget")
    end
  end

  # Validate only the DiT encode interval. The caller's receipt hash may also
  # contain nonzero counts from the already-drained reference encoder phase.
  def self.validate_dit_encode_metrics!(metrics : Hash(String, Int64)) : Nil
    DIT_ENCODE_METRIC_KEYS.each do |key|
      value = metrics[key]? || raise ArgumentError.new("Missing DiT encode metric #{key}")
      unless value == 0_i64
        raise ArgumentError.new("DiT encode introduced allocation/readback/wait: #{key}=#{value}")
      end
    end
  end

  def self.validate_encoder_sha!(sha : String) : String
    unless {ENCODER_SAFETENSORS_SHA256, RECOVERED_ENCODER_SAFETENSORS_SHA256}.includes?(sha)
      raise ArgumentError.new("reference encoder safetensors SHA-256 is not an admitted encoder identity")
    end
    sha
  end

  def self.validate_encoder_identity!(requested_sha : String, inputs_sha : String) : Nil
    validate_encoder_sha!(requested_sha)
    unless inputs_sha == requested_sha
      raise ArgumentError.new("reference image encoder SHA-256 differs from the requested encoder identity")
    end
  end

  # Pure command-line contract so the all-or-none and unqualified execution
  # gate can be exercised without opening Metal or authenticating model files.
  def self.validate_cli!(image_inputs_path : String, image_inputs_sha : String,
                         encoder_path : String, encoder_sha : String, *,
                         allow_unqualified : Bool, plan_only : Bool,
                         cinematic_requested : Bool, legacy_text : Bool) : Bool
    fields = {image_inputs_path, image_inputs_sha, encoder_path, encoder_sha}
    present = fields.count { |value| !value.empty? }
    unless present == 0 || present == fields.size
      raise ArgumentError.new("reference image inputs and encoder path/SHA flags must be supplied together")
    end
    requested = present == fields.size
    if allow_unqualified && !requested
      raise ArgumentError.new("--allow-unqualified-reference requires all reference image and encoder flags")
    end
    validate_encoder_sha!(encoder_sha) if requested
    if requested && cinematic_requested
      raise ArgumentError.new("reference initialization does not support Cinematic checkpoints")
    end
    if requested && legacy_text
      raise ArgumentError.new("reference initialization requires once-per-scene text caching")
    end
    if requested && !plan_only && !allow_unqualified
      raise ArgumentError.new("reference execution is unqualified; pass --allow-unqualified-reference to opt in")
    end
    requested
  end

  # One setup/encode/DiT-borrow lifetime for the single T0 reference. The
  # constructor does only CPU authentication and planning. Metal owners are
  # created only by prepare! after CLI preflight and plan-only handling.
  class Session
    getter residency : ResidencyPlan
    getter named_bytes : Int64
    getter metrics : NamedTuple(allocations: Int64, readbacks: Int64, preterminal_waits: Int64)?

    @inputs : ML::Video::HyNomasReferenceImage::Inputs
    @config : ML::Video::DCAEVEncoderConfig
    @budget_bytes : Int64
    @weights_file : ML::Video::CheckpointFile?
    @weight_plan : ML::Video::WeightPlan?
    @weights_arena : ML::Video::Arena? = nil
    @workspace_arena : ML::Video::Arena? = nil
    @clip_arena : ML::Video::Arena? = nil
    @latent_arena : ML::Video::Arena? = nil
    @workspace : ML::Video::DCAEVEncoderWorkspace? = nil
    @mps_plans : ML::Video::DCAEVEncoderMPSPlanSet? = nil
    @prepared : ML::Video::PreparedWeights? = nil
    @rgb : ML::MetalBuffer? = nil
    @clip : ML::MetalBuffer? = nil
    @latent : ML::MetalBuffer? = nil
    @encoder_chain : ML::Metal::MPSCommandChain? = nil
    @encoder_chain_id : UInt64? = nil
    @dit_chain : ML::Metal::MPSCommandChain? = nil
    @dit_chain_id : UInt64? = nil
    @prepared_for_encode = false
    @encode_started = false
    @encoder_phase_released = false
    @closed = false

    def initialize(@inputs : ML::Video::HyNomasReferenceImage::Inputs,
                   weights_path : String, weights_sha256 : String,
                   @budget_bytes : Int64)
      HyNomasReferenceEncoder.validate_encoder_identity!(weights_sha256, @inputs.encoder_sha)
      raise ArgumentError.new("reference encoder budget must be positive") unless @budget_bytes > 0
      @config = FIXED_CFG
      @residency = ResidencyPlan.for(@config)
      @named_bytes = @residency.named_bytes
      HyNomasReferenceEncoder.validate_budget!(@residency, @budget_bytes)

      # The image helper authenticates schema, source provenance, and the
      # prepared RGB payload. Recheck here so CPU preflight binds the exact
      # bytes this session will upload, without requiring the source image to
      # remain available after preparation.
      @inputs.verify_payload!
      weights_file = ML::Video::BF16WeightCache.authenticate(weights_path, weights_sha256)
      begin
        bindings = ML::Video::DCAEVEncoder.checkpoint_bindings(@config)
        weight_plan = ML::Video::WeightPlan.new([weights_file], bindings, @budget_bytes,
          reserve_bytes: @named_bytes - @residency.weight_bytes,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        unless weight_plan.staging_bytes == 0 &&
               weight_plan.entries.size == bindings.size &&
               weight_plan.destination_bytes == @residency.weight_bytes &&
               weight_plan.peak_bytes == @named_bytes && bindings.size == 169
          raise ArgumentError.new("reference encoder weight inventory or residency plan differs from the fixed DCAE contract")
        end
        @weight_plan = weight_plan
        @weights_file = weights_file
      rescue ex
        weights_file.close
        raise ex
      end
    end

    # Allocate the exact peak and upload the authenticated one-frame RGB8
    # payload. No input clip or encoder latent is read back to the CPU.
    def prepare! : Nil
      raise ArgumentError.new("reference encoder session is closed") if @closed
      raise ArgumentError.new("reference encoder session is already prepared") if @prepared_for_encode
      @inputs.verify_payload!

      begin
        @weights_arena = ML::Video::Arena.new
        @workspace_arena = ML::Video::Arena.new
        @clip_arena = ML::Video::Arena.new
        @latent_arena = ML::Video::Arena.new
        @prepared = ML::Video::PreparedWeights.new(@weight_plan.not_nil!, @weights_arena.not_nil!)
        @workspace = ML::Video::DCAEVEncoder.prepare!(@workspace_arena.not_nil!, @config,
          @budget_bytes, @named_bytes - @residency.workspace_bytes)
        @clip = @clip_arena.not_nil!.alloc_weight("dcae_v.reference.clip",
          @config.input_values, ML::Video::WeightDType::BF16)
        @latent = @latent_arena.not_nil!.alloc_weight("dcae_v.reference.latent",
          @config.latent_values, ML::Video::WeightDType::BF16)
        @rgb = ML::MetalBuffer.new(@config.rgb_bytes)
        @inputs.upload!(@rgb.not_nil!)

        actual = @weights_arena.not_nil!.resident_bytes +
                 @workspace_arena.not_nil!.resident_bytes +
                 @clip_arena.not_nil!.resident_bytes +
                 @latent_arena.not_nil!.resident_bytes + @rgb.not_nil!.size
        unless actual == @named_bytes && actual <= @budget_bytes
          raise ArgumentError.new("reference encoder allocation differs from its admitted named residency")
        end
        @mps_plans = ML::Video::DCAEVEncoder.prepare_mps!(@clip.not_nil!, @latent.not_nil!,
          @weights_arena.not_nil!, @workspace.not_nil!, @config)
        @prepared_for_encode = true
      rescue ex
        release_setup_owners!
        raise ex
      end
    end

    # Enqueue one normalize/repeat kernel followed by DCAE-V. All Metal
    # allocations, host uploads, and pipeline preparation happen before this
    # method; only the caller performs the terminal wait.
    def encode!(chain : ML::Metal::MPSCommandChain) : NamedTuple(allocations: Int64, readbacks: Int64, preterminal_waits: Int64)
      raise ArgumentError.new("reference encoder session is not ready") unless @prepared_for_encode && @mps_plans && !@closed
      raise ArgumentError.new("reference encoder may run only once") if @encode_started
      unless !chain.drained? && chain.wait_count == 0
        raise ArgumentError.new("reference encoder requires a fresh undrained command chain")
      end
      @encode_started = true
      @encoder_chain = chain
      @encoder_chain_id = chain.object_id
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks
      ML::Video::DCAEVEncoder.encode_image_pixels!(chain.encoder, @rgb.not_nil!,
        @clip.not_nil!, @config)
      ML::Video::DCAEVEncoder.encode(chain.encoder, @clip.not_nil!, @latent.not_nil!,
        @weights_arena.not_nil!, @workspace.not_nil!, @config, chain, nil,
        @mps_plans.not_nil!)
      @metrics = {
        allocations:       ML::MetalBuffer.hy_text_probe_allocations - allocations,
        readbacks:         ML::MetalBuffer.hy_text_probe_readbacks - readbacks,
        preterminal_waits: chain.wait_count.to_i64,
      }
      unless @metrics.not_nil!.values.all?(&.zero?)
        raise "reference encoder encode allocated, read back, or waited before terminal drain"
      end
      @metrics.not_nil!
    end

    # Called only after the single encoder chain terminal drain. Free phase
    # inputs and weights, keeping the caller-owned BF16 latent resident for DiT.
    def finish_encoder_phase!(chain : ML::Metal::MPSCommandChain) : ML::MetalBuffer
      validate_encoder_drain!(chain)
      return @latent.not_nil! if @encoder_phase_released
      release_encoder_phase_owners!(chain)
      @encoder_phase_released = true
      @encoder_chain = nil
      @latent.not_nil!
    end

    def latent_for_dit : ML::MetalBuffer
      raise ArgumentError.new("reference latent is unavailable before encoder phase completion") unless @encoder_phase_released && !@closed
      @latent || raise ArgumentError.new("reference latent has been released")
    end

    # Tie the latent's retained lifetime to the one DiT chain that consumes it.
    def bind_dit_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      latent_for_dit
      raise ArgumentError.new("reference latent is already bound to a DiT chain") if @dit_chain_id
      unless !chain.drained? && chain.wait_count == 0 && @encoder_chain_id != chain.object_id
        raise ArgumentError.new("reference latent requires a fresh DiT chain after the encoder drain")
      end
      @dit_chain = chain
      @dit_chain_id = chain.object_id
    end

    # Release the latent only after DiT has completed its terminal wait and
    # released its schedule/borrowed input owners.
    def release_after_dit_drain!(chain : ML::Metal::MPSCommandChain) : Nil
      unless @dit_chain_id == chain.object_id && chain.drained? && chain.wait_count == 1
        raise ArgumentError.new("reference latent release requires its DiT terminal drain")
      end
      release_latent_owner!
      @dit_chain = nil
      @dit_chain_id = nil
      @encoder_chain = nil
      @encoder_chain_id = nil
      @closed = true
      @weights_file.try(&.close)
      @weights_file = nil
    end

    # Safe plan-only or error cleanup. If an encode/DiT chain exists, its
    # terminal drain is required before any corresponding GPU owner is freed.
    def close! : Nil
      return if @closed
      encoder_chain = @encoder_chain
      if dit_chain = @dit_chain
        unless dit_chain.drained? && dit_chain.wait_count == 1
          raise ArgumentError.new("reference session cannot close before the DiT terminal drain")
        end
      elsif encoder_chain = @encoder_chain
        unless encoder_chain.drained? && encoder_chain.wait_count == 1
          raise ArgumentError.new("reference session cannot close before the encoder terminal drain")
        end
      end
      release_encoder_phase_owners!(encoder_chain) unless @encoder_phase_released
      release_latent_owner!
      @weights_file.try(&.close)
      @weights_file = nil
      @dit_chain = nil
      @dit_chain_id = nil
      @encoder_chain = nil
      @encoder_chain_id = nil
      @closed = true
    end

    private def validate_encoder_drain!(chain : ML::Metal::MPSCommandChain) : Nil
      unless @encode_started && @encoder_chain_id == chain.object_id &&
             chain.drained? && chain.wait_count == 1
        raise ArgumentError.new("reference latent admission requires the encoder's single terminal drain")
      end
    end

    private def release_encoder_phase_owners!(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @encoder_phase_released
      release_setup_owners!(chain)
      @weights_file.try(&.close)
      @weights_file = nil
      @weight_plan = nil
    end

    private def release_setup_owners!(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      if plans = @mps_plans
        plans.release(chain)
        @mps_plans = nil
      end
      @prepared.try(&.release)
      @prepared = nil
      @rgb.try(&.release)
      @rgb = nil
      @clip_arena.try(&.release)
      @clip_arena = nil
      @workspace_arena.try(&.release)
      @workspace_arena = nil
      @weights_arena.try(&.release)
      @weights_arena = nil
      @workspace = nil
      @prepared_for_encode = false if !@encode_started
    end

    private def release_latent_owner! : Nil
      @latent_arena.try(&.release)
      @latent_arena = nil
      @latent = nil
    end
  end
end
