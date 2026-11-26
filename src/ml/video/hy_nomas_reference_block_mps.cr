# Opt-in mixed-precision reference diagnostic. Requiring ordinary block_mps
# does not pull the BF16 linear bridge into existing text/video executables.
require "./hy_nomas_block_mps"
require "./hy_nomas_mps_bf16_linear"
require "./hy_nomas_reference_bf16"
require "./hy_nomas_reference_kv_cache"
require "./hy_nomas_reference_attention"

module ML::Video
  # Fixed bindings for source BF16 boundaries up to prospective joint K/V.
  # An optional first-T0 BF16 cache and attention consumer stop before residual
  # and FFN. History append and complete block execution remain unqualified.
  class HyNomasReferenceMPSWorkspace < HyNomasMPSWorkspace
    @reference_arena : Arena? = nil
    @reference_source_inputs = [] of ML::MetalBuffer
    @reference_inputs = [] of ML::MetalBuffer
    @reference_linear_plans = {} of String => HyNomasMPSBF16Linear::Plan
    @reference_activations = {} of String => HyNomasMPSActivation::SiLUPlan
    @reference_activation_operations = {} of String => HyNomasDualStreamBlock::MPSActivationOperation
    @reference_pointwise : HyNomasReferenceBF16::Plan? = nil
    @reference_history_cache : HyNomasReferenceKVCache::Plan? = nil
    @reference_attention_plan : HyNomasReferenceAttention::Plan? = nil
    @reference_history_attempted = false
    @reference_released = false

    def reference_preattention_bf16 : Bool
      true
    end

    protected def source_bf16_rms? : Bool
      true
    end

    def self.validate_reference_config!(config : HyNomasConfig) : Nil
      config.validate!
      unless config.batches == 1 && config.effective_img_modulation_rows == 1 &&
             config.effective_txt_modulation_rows == 1 &&
             config.attention_precision.bf16_boundary_control?
        raise ArgumentError.new("Reference BF16 preattention requires batch-one global modulation and BF16 boundary packing")
      end
    end

    # Added allocations only: four rounded F32 input copies, plus physical
    # BF16 outputs for two modulation and six Q/K/V projections. RMS and the
    # unused ordinary attention/linear plans retain their base accounting.
    def self.reference_extra_bytes(config : HyNomasConfig) : Int64
      validate_reference_config!(config)
      stream_elements = config.image_rows.to_i64 * config.dim + config.text_rows.to_i64 * config.dim
      modulation_elements = (config.effective_img_modulation_rows.to_i64 + config.effective_txt_modulation_rows) * config.dim
      (stream_elements + modulation_elements) * 4_i64 +
        (3_i64 * stream_elements + 6_i64 * modulation_elements) * 2_i64
    end

    def initialize(block : HyNomasDualStreamBlock,
                   img : ML::MetalBuffer, txt : ML::MetalBuffer,
                   vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                   video_rope_cos : ML::MetalBuffer,
                   video_rope_sin : ML::MetalBuffer,
                   *, attention_query_chunk : Int32 = 0,
                   attention_heads_per_group : Int32 = 0,
                   allow_shared_modulation : Bool = false,
                   valid_text_tokens : Int32? = nil,
                   text_rms_tokens : Int32? = nil,
                   reference_history : Bool = false,
                   reference_attention : Bool = false,
                   reference_linear_variant : HyNomasMPSBF16Linear::Variant = HyNomasMPSBF16Linear::Variant::BF16GraphOps)
      self.class.validate_reference_config!(block.config)
      unless {HyNomasMPSBF16Linear::Variant::BF16GraphOps,
              HyNomasMPSBF16Linear::Variant::F32Accumulation}.includes?(reference_linear_variant)
        raise ArgumentError.new("Unknown reference BF16 linear variant")
      end
      if reference_history && reference_linear_variant != HyNomasMPSBF16Linear::Variant::BF16GraphOps
        raise ArgumentError.new("Reference BF16 history requires the BF16 graph linear variant")
      end
      if reference_attention && !reference_history
        raise ArgumentError.new("Reference BF16 attention requires physical history storage")
      end
      operations = block.mps_linear_operations.select { |operation| reference_linear?(operation.name) }
      unless operations.size == 8 && operations.all? { |operation| operation.weights.weight_dtype.bf16? && operation.weights.bias_dtype.bf16? }
        raise ArgumentError.new("Reference BF16 preattention requires eight BF16 projection weights and biases")
      end
      @reference_source_inputs = [img, txt, vec, vec_txt]
      super(block, img, txt, vec, vec_txt, video_rope_cos, video_rope_sin,
        attention_query_chunk: attention_query_chunk,
        attention_heads_per_group: attention_heads_per_group,
        allow_shared_modulation: allow_shared_modulation,
        valid_text_tokens: valid_text_tokens, text_rms_tokens: text_rms_tokens)
      begin
        arena = Arena.new
        @reference_arena = arena
        @reference_pointwise = HyNomasReferenceBF16::Plan.new(config.dim)
        @reference_source_inputs.each_with_index do |input, index|
          @reference_inputs << arena.alloc("reference.rounded_input.#{index}", (input.size // 4).to_i32)
        end
        operations.each do |operation|
          count = HyNomasDualStreamBlock.checked_count(operation.rows, operation.output_dim)
          raw = arena.alloc_weight("reference.#{operation.name}.bf16", count, WeightDType::BF16)
          @reference_linear_plans[operation.name] = HyNomasMPSBF16Linear::Plan.new(
            operation.input, operation.weights, raw, operation.output,
            reference_linear_variant, operation.rows, operation.input_dim, operation.output_dim)
        end
        block.mps_activation_operations(vec, vec_txt).each do |operation|
          next unless operation.name.ends_with?(".silu")
          @reference_activation_operations[operation.name] = operation
          index = operation.name.starts_with?("img.") ? 2 : 3
          @reference_activations[operation.name] = HyNomasMPSActivation::SiLUPlan.new(
            operation.shape, @reference_inputs[index], operation.output)
        end
        unless arena.resident_bytes == self.class.reference_extra_bytes(config)
          raise ArgumentError.new("Reference BF16 allocation accounting changed")
        end
        if reference_history
          @reference_history_cache = HyNomasReferenceKVCache::Plan.new(
            block.checkpoints["joint.key"], block.checkpoints["joint.value"],
            storage_rows: config.joint_tokens, active_rows: attention_tokens,
            heads: config.heads, head_dim: config.head_dim)
          if reference_attention
            @reference_attention_plan = HyNomasReferenceAttention::Plan.new(
              block.checkpoints["joint.query"], @reference_history_cache.not_nil!,
              block.checkpoints["joint.attention"])
          end
        end
      rescue ex
        release
        raise ex
      end
    end

    def resident_bytes : Int64
      super + (@reference_arena.try(&.resident_bytes) || 0_i64) +
        (@reference_history_cache.try(&.resident_bytes) || 0_i64) +
        (@reference_attention_plan.try(&.resident_bytes) || 0_i64)
    end

    def reference_history_cache : HyNomasReferenceKVCache::Plan
      raise ArgumentError.new("Reference BF16 workspace has been released") if @reference_released
      @reference_history_cache || raise ArgumentError.new("Reference BF16 history cache was not prepared")
    end

    def reference_history_prepared? : Bool
      !@reference_history_cache.nil?
    end

    def reference_attention_plan : HyNomasReferenceAttention::Plan
      raise ArgumentError.new("Reference BF16 workspace has been released") if @reference_released
      @reference_attention_plan || raise ArgumentError.new("Reference BF16 attention was not prepared")
    end

    def encode_reference_attention(chain : ML::Metal::MPSCommandChain) : ML::MetalBuffer
      reference_attention_plan.encode(chain)
    end

    # A partial projection failure already leaves queued work on the chain.
    # Consume this lease before the first projection, even if no cache write
    # is reached. Input and cache preflight failures do not consume the lease.
    def claim_reference_history_store!(chain : ML::Metal::MPSCommandChain) : HyNomasReferenceKVCache::Plan
      cache = reference_history_cache
      cache.validate_store!(chain)
      raise ArgumentError.new("Reference BF16 history attempt already started") if @reference_history_attempted
      @reference_history_attempted = true
      cache
    end

    def project(chain : ML::Metal::MPSCommandChain, name : String,
                input : ML::MetalBuffer, weights : HyNomasDualStreamBlock::LinearWeights,
                output : ML::MetalBuffer, rows : Int32,
                input_dim : Int32, output_dim : Int32) : Nil
      validate_projection!(chain, name, input, weights, output, rows, input_dim, output_dim)
      plan = @reference_linear_plans[name]? || raise ArgumentError.new("Reference BF16 preattention rejects projection #{name}")
      plan.encode(chain)
    end

    def activation(chain : ML::Metal::MPSCommandChain, name : String,
                   input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      operation = @reference_activation_operations[name]?
      unless operation && operation.input.same?(input) && operation.output.same?(output)
        raise ArgumentError.new("Reference BF16 preattention SiLU binding changed")
      end
      plan = @reference_activations[name]? || raise ArgumentError.new("Reference BF16 SiLU plan is unavailable")
      index = name.starts_with?("img.") ? 2 : 3
      count = (input.size // 4).to_i32
      pointwise.round(chain.encoder, input, @reference_inputs[index], count)
      plan.encode(chain)
      pointwise.round(chain.encoder, output, output, count)
    end

    def layer_norm(e : ML::Metal::ComputeEncoder,
                   input : ML::MetalBuffer, output : ML::MetalBuffer,
                   rows : Int32, dim : Int32, epsilon : Float32) : Nil
      index = @reference_source_inputs.index { |candidate| candidate.same?(input) }
      unless index && index < 2
        raise ArgumentError.new("Reference BF16 preattention LayerNorm input changed")
      end
      rounded = @reference_inputs[index]
      pointwise.round(e, input, rounded, HyNomasDualStreamBlock.checked_count(rows, dim))
      super(e, rounded, output, rows, dim, epsilon)
    end

    def modulate(e : ML::Metal::ComputeEncoder,
                 input : ML::MetalBuffer, shift : ML::MetalBuffer,
                 scale : ML::MetalBuffer, output : ML::MetalBuffer,
                 rows : Int32, dim : Int32) : Nil
      validate_pointwise_geometry!(rows, dim)
      validate_pointwise_bindings!(input, shift, scale, output)
      pointwise.modulate(e, input, shift, scale, output, rows, dim)
    end

    def rope(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
             output : ML::MetalBuffer, cosine : ML::MetalBuffer,
             sine : ML::MetalBuffer, tokens : Int32) : Nil
      super
      pointwise.round(e, output, output, HyNomasDualStreamBlock.checked_count(config.batches, tokens, config.dim))
    end

    def release : Nil
      return if @reference_released
      if chain = @last_chain
        raise ArgumentError.new("Complete reference BF16 chain before releasing workspace") unless chain.completed?
      end
      # Preflight every nested lease before freeing any resource: cache reads
      # or attention can have a newer chain than the preattention workspace.
      @reference_history_cache.try(&.validate_release!)
      @reference_attention_plan.try(&.validate_release!)
      @reference_attention_plan.try(&.release)
      @reference_attention_plan = nil
      @reference_history_cache.try(&.release)
      @reference_history_cache = nil
      @reference_activations.each_value(&.release)
      @reference_activations.clear
      @reference_linear_plans.each_value(&.release)
      @reference_linear_plans.clear
      @reference_arena.try(&.release)
      @reference_arena = nil
      super
      @reference_released = true
    end

    private def pointwise : HyNomasReferenceBF16::Plan
      raise ArgumentError.new("Reference BF16 workspace has been released") if @reference_released
      @reference_pointwise || raise ArgumentError.new("Reference BF16 pointwise plan is unavailable")
    end

    private def reference_linear?(name : String) : Bool
      name.ends_with?(".modulation") || name.ends_with?(".attn.q") ||
        name.ends_with?(".attn.k") || name.ends_with?(".attn.v")
    end
  end

  class HyNomasDualStreamBlock
    # Bounded first-T0 consumer: preserve the single-store preflight/latch,
    # then attend directly to the physical BF16 cache on the same GPU chain.
    # Returning context does not admit the rest of the dual-stream block.
    def encode_mps_reference_attention(chain : ML::Metal::MPSCommandChain,
                                       img : ML::MetalBuffer, txt : ML::MetalBuffer,
                                       vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                                       video_rope_cos : ML::MetalBuffer,
                                       video_rope_sin : ML::MetalBuffer,
                                       workspace : HyNomasReferenceMPSWorkspace) : ML::MetalBuffer
      workspace.reference_attention_plan # Reject an unprepared consumer before projection.
      encode_mps_reference_history(chain, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin, workspace)
      workspace.encode_reference_attention(chain)
    end

    # First-T0 qualification only: project current inputs, persist the active
    # joint K/V prefix as BF16, and read that physical cache into F32 stages.
    # Preflight the single-write lease before appending any projection work.
    def encode_mps_reference_history(chain : ML::Metal::MPSCommandChain,
                                     img : ML::MetalBuffer, txt : ML::MetalBuffer,
                                     vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                                     video_rope_cos : ML::MetalBuffer,
                                     video_rope_sin : ML::MetalBuffer,
                                     workspace : HyNomasReferenceMPSWorkspace) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      workspace.validate_for!(self, chain, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin, bind_chain: false)
      cache = workspace.claim_reference_history_store!(chain)
      backend = HyNomasMPSBackend.new(chain, workspace)
      encode_with_backend(chain.encoder, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin, backend, stop_before_attention: true)
      cache.store(chain)
      cache.read(chain)
      {cache.key_read, cache.value_read}
    end

    # The returned buffers are prospective concatenated K/V, never image/text
    # block outputs. No attention, history write or continuation is encoded.
    def encode_mps_preattention(chain : ML::Metal::MPSCommandChain,
                                img : ML::MetalBuffer, txt : ML::MetalBuffer,
                                vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                                video_rope_cos : ML::MetalBuffer,
                                video_rope_sin : ML::MetalBuffer,
                                workspace : HyNomasMPSWorkspace) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      unless workspace.reference_preattention_bf16
        raise ArgumentError.new("Reference preattention requires a BF16 reference workspace")
      end
      if workspace.is_a?(HyNomasReferenceMPSWorkspace) && workspace.reference_history_prepared?
        raise ArgumentError.new("Reference BF16 history workspace requires encode_mps_reference_history")
      end
      backend = HyNomasMPSBackend.new(chain, workspace)
      encode_with_backend(chain.encoder, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin, backend, stop_before_attention: true)
    end
  end
end
