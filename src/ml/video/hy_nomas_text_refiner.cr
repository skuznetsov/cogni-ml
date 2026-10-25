# Native Metal/MPS execution for the pinned single-token Hy-Nomas text refiner.
# The implementation keeps the Qwen input in BF16, compacts the immutable CPU
# prefix mask on the GPU, and runs the refiner in F32 unless attention boundary
# rounding is explicitly selected.
require "../gguf/safetensors"
require "../metal/dispatch"
require "./checkpoint"
require "./metal_ops"
require "./metal_mps_command_chain"
require "./metal_linear_mps"
require "./hy_nomas_attention_mps"
require "./hy_nomas_mps_norm"
require "./hy_nomas_mps_activation"

module ML::Video
  class HyNomasTextRefinerMPSPlan
    TOKEN_COUNT  = 1280
    INPUT_DIM    = 3584
    HIDDEN_DIM   = 2048
    HEADS        =   16
    HEAD_DIM     =  128
    FFN_DIM      = 8192
    TIMESTEP_DIM =  256
    BLOCKS       =    2

    COMPACT_KERNEL  = "hy_nomas_text_compact_bf16_f32"
    MEAN_KERNEL     = "hy_nomas_text_masked_mean_f32"
    TIME_KERNEL     = "hy_nomas_text_timestep_sinusoidal_f32"
    SPLIT_KERNEL    = "hy_nomas_text_qkv_split_f32"
    ADALN_KERNEL    = "hy_nomas_text_split_adaln_f32"
    RESIDUAL_KERNEL = "hy_nomas_text_gated_residual_f32"
    AFFINE_KERNEL   = "hy_nomas_text_layer_norm_affine_f32"
    ROUND_KERNEL    = "hy_nomas_text_roundtrip_bf16_f32"

    SOURCE = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_text_refiner.metal") }}

    record NamedBytes, activations : Int64, linear_stage : Int64,
      attention_stage : Int64, total : Int64

    private record LinearEntry, input : ML::MetalBuffer, output : ML::MetalBuffer,
      weight : ML::MetalBuffer, bias : ML::MetalBuffer,
      weight_dtype : WeightDType, bias_dtype : WeightDType,
      plan : WanT5MPSLinearPlan, rows : Int32, input_dim : Int32,
      output_dim : Int32

    # Keep native plans rooted until explicit release after the terminal wait.
    # This mirrors the Qwen conditioning/MPS workspace lease: abandoning a live
    # plan leaks safely rather than allowing a finalizer to invalidate GPU work.
    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of HyNomasTextRefinerMPSPlan

    @source : ML::MetalBuffer
    @mask : Array(Int32)
    @valid_tokens : Int32
    @timestep : Float32
    @weights : Arena
    @attention_precision : HyNomasAttentionPrecision
    @named_bytes : NamedBytes
    @arena : Arena? = nil
    @weight_stage : ML::MetalBuffer? = nil
    @checkpoints = {} of String => ML::MetalBuffer
    @linears = {} of String => LinearEntry
    @linear_plans = [] of WanT5MPSLinearPlan
    @activations = {} of String => HyNomasMPSActivation::UnaryPlan
    @activation_plans = [] of HyNomasMPSActivation::UnaryPlan
    @attention_plans = [] of HyNomasAttentionMPSPlan
    @norm_plan : HyNomasMPSLayerNorm::Plan? = nil
    @pipelines = {} of String => ML::Metal::ComputePipeline
    @external_weights = [] of Tuple(String, ML::MetalBuffer, WeightDType, Int64)
    @owned_bindings = [] of Tuple(ML::MetalBuffer, Int64)
    @all_external = [] of ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    getter valid_tokens : Int32
    getter admitted_named_bytes : NamedBytes
    getter attention_precision : HyNomasAttentionPrecision

    def self.checkpoint_bindings : Array(WeightBinding)
      bindings = [] of WeightBinding
      append_linear_bindings(bindings, "txt_in.input_embedder", INPUT_DIM, HIDDEN_DIM)
      append_linear_bindings(bindings, "txt_in.t_embedder.mlp.0", TIMESTEP_DIM, HIDDEN_DIM)
      append_linear_bindings(bindings, "txt_in.t_embedder.mlp.2", HIDDEN_DIM, HIDDEN_DIM)
      append_linear_bindings(bindings, "txt_in.c_embedder.linear_1", INPUT_DIM, HIDDEN_DIM)
      append_linear_bindings(bindings, "txt_in.c_embedder.linear_2", HIDDEN_DIM, HIDDEN_DIM)
      BLOCKS.times do |block|
        prefix = "txt_in.individual_token_refiner.blocks.#{block}"
        append_vector_bindings(bindings, "#{prefix}.norm1", HIDDEN_DIM)
        append_linear_bindings(bindings, "#{prefix}.self_attn_qkv", HIDDEN_DIM, 3 * HIDDEN_DIM)
        append_linear_bindings(bindings, "#{prefix}.self_attn_proj", HIDDEN_DIM, HIDDEN_DIM)
        append_vector_bindings(bindings, "#{prefix}.norm2", HIDDEN_DIM)
        append_linear_bindings(bindings, "#{prefix}.mlp.fc1", HIDDEN_DIM, FFN_DIM)
        append_linear_bindings(bindings, "#{prefix}.mlp.fc2", FFN_DIM, HIDDEN_DIM)
        append_linear_bindings(bindings, "#{prefix}.adaLN_modulation.1", HIDDEN_DIM, 2 * HIDDEN_DIM)
      end
      append_linear_bindings(bindings, "time_in.mlp.0", TIMESTEP_DIM, HIDDEN_DIM)
      append_linear_bindings(bindings, "time_in.mlp.2", HIDDEN_DIM, HIDDEN_DIM)
      bindings
    end

    def self.valid_tokens(mask : Array(Int32)) : Int32
      unless mask.size == TOKEN_COUNT
        raise ArgumentError.new("Hy-Nomas text mask must have exactly #{TOKEN_COUNT} tokens")
      end
      valid = 0_i32
      mask.each do |value|
        unless value == 0 || value == 1
          raise ArgumentError.new("Hy-Nomas text mask must be binary")
        end
        valid += 1 if value == 1
      end
      raise ArgumentError.new("Hy-Nomas text mask must contain a valid token") if valid == 0
      unless mask[0] == 1
        raise ArgumentError.new("Hy-Nomas text mask must keep token zero valid")
      end
      saw_padding = false
      mask.each do |value|
        if value == 0
          saw_padding = true
        else
          raise ArgumentError.new("Hy-Nomas text mask must be one contiguous valid prefix") if saw_padding
        end
      end
      valid
    end

    def self.named_bytes(mask : Array(Int32)) : NamedBytes
      valid = valid_tokens(mask).to_i64
      # Every unique F32 activation is listed once. residual2/output and
      # projected_text are checkpoint aliases, not extra storage.
      fixed_elements = 24_321_i64 + 20_480_i64
      token_elements = 5_632_i64 + 2_i64 * 43_008_i64
      activation = (fixed_elements + token_elements * valid) * sizeof(Float32)
      linear_stage = HIDDEN_DIM.to_i64 * FFN_DIM * sizeof(Float32)
      # Each block owns an attention plan with four fixed head-major F32
      # staging buffers (Q, K, V, output).
      attention_stage = BLOCKS.to_i64 * 4_i64 * valid * HIDDEN_DIM * sizeof(Float32)
      NamedBytes.new(activation, linear_stage, attention_stage,
        activation + linear_stage + attention_stage)
    end

    def self.admit_named_bytes!(mask : Array(Int32), max_named_bytes : Int64,
                                seg_pos_emb_num : Int32 = 0) : NamedBytes
      unless seg_pos_emb_num == 0
        raise ArgumentError.new("Hy-Nomas text refiner requires seg_pos_emb_num=0")
      end
      bytes = named_bytes(mask)
      unless max_named_bytes > 0 && bytes.total <= max_named_bytes
        raise ArgumentError.new("Hy-Nomas text refiner exceeds admitted named-buffer budget")
      end
      bytes
    end

    def initialize(source : ML::MetalBuffer, mask : Array(Int32), timestep : Float32,
                   weights : Arena, max_named_bytes : Int64,
                   attention_precision : HyNomasAttentionPrecision = HyNomasAttentionPrecision::F32Control,
                   *, seg_pos_emb_num : Int32 = 0)
      @source = source
      @mask = mask.dup
      @valid_tokens = self.class.valid_tokens(@mask)
      @timestep = timestep
      @weights = weights
      @attention_precision = attention_precision
      @named_bytes = self.class.admit_named_bytes!(@mask, max_named_bytes, seg_pos_emb_num)
      @admitted_named_bytes = @named_bytes
      unless @timestep.finite?
        raise ArgumentError.new("Hy-Nomas text timestep must be finite")
      end
      unless @attention_precision.f32_control? || @attention_precision.bf16_boundary_control?
        raise ArgumentError.new("Unsupported Hy-Nomas text attention precision control")
      end

      validate_source!
      validate_weights!
      @all_external << @source
      @external_weights.each { |_, weight, _, _| @all_external << weight }

      @arena = Arena.new
      begin
        allocate_buffers!
        prepare_pipelines!
        prepare_plans!
        validate_bindings!
        unless resident_bytes == @named_bytes.total
          raise ArgumentError.new("Hy-Nomas text refiner named-buffer accounting mismatch")
        end
        retain_lifetime_root!
      rescue ex
        release_prepared_plans
        arena.release
        @released = true
        raise ex
      end
    end

    def self.retained_plan_count : Int32
      @@lifetime_mutex.synchronize { @@live_plans.size.to_i32 }
    end

    def checkpoints : Hash(String, ML::MetalBuffer)
      ensure_live!
      @checkpoints.dup
    end

    def text_output : ML::MetalBuffer
      ensure_live!
      @checkpoints["projected_text"]
    end

    def vec : ML::MetalBuffer
      ensure_live!
      @checkpoints["vec"]
    end

    def vec_txt : ML::MetalBuffer
      vec
    end

    def resident_bytes : Int64
      ensure_live!
      arena.resident_bytes + @attention_plans.sum(&.resident_bytes)
    end

    # Append the complete compaction, two timestep paths, context projection,
    # and two refiner blocks to the caller's existing serial MPS command chain.
    # Every pipeline, buffer and graph is setup-owned; this method never
    # allocates, waits, reads a GPU buffer, or submits a command.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      ensure_live!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas text refiner cannot span in-flight command chains")
        end
      end
      validate_bindings!
      @last_chain = chain

      encode_compaction(chain.encoder)
      encode_mean(chain.encoder)
      encode_sinusoidal(chain.encoder)
      encode_linear(chain, "refiner_timestep_mlp1")
      encode_activation(chain, "refiner_timestep_silu")
      encode_linear(chain, "refiner_timestep")
      encode_linear(chain, "context_linear1")
      encode_activation(chain, "context_silu")
      encode_linear(chain, "context_embed")
      MetalOps.add(chain.encoder, checkpoint("refiner_timestep"), checkpoint("context_embed"),
        checkpoint("c"), HIDDEN_DIM)
      encode_linear(chain, "input_embed")
      encode_linear(chain, "time_in.mlp1")
      encode_activation(chain, "time_in.silu")
      encode_linear(chain, "time_in.mlp2")

      BLOCKS.times do |block|
        encode_block(chain, block)
      end
    end

    # Release the strong retention root and all prepared plans only after their
    # final command chain has completed its existing terminal wait.
    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas text refiner command chain before release") unless chain.completed?
      end
      release_prepared_plans
      arena.release
      @arena = nil
      @weight_stage = nil
      @checkpoints.clear
      @linears.clear
      @released = true
      release_lifetime_root!
    end

    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end

    private def self.append_linear_bindings(bindings : Array(WeightBinding), prefix : String,
                                            input_dim : Int32, output_dim : Int32) : Nil
      bindings << WeightBinding.new("#{prefix}.weight", "#{prefix}.weight",
        [output_dim.to_i64, input_dim.to_i64], ML::GGUF::SafeTensorDType::F32)
      bindings << WeightBinding.new("#{prefix}.bias", "#{prefix}.bias",
        [output_dim.to_i64], ML::GGUF::SafeTensorDType::F32)
    end

    private def self.append_vector_bindings(bindings : Array(WeightBinding), prefix : String,
                                            dim : Int32) : Nil
      bindings << WeightBinding.new("#{prefix}.weight", "#{prefix}.weight",
        [dim.to_i64], ML::GGUF::SafeTensorDType::F32)
      bindings << WeightBinding.new("#{prefix}.bias", "#{prefix}.bias",
        [dim.to_i64], ML::GGUF::SafeTensorDType::F32)
    end

    private def ensure_live! : Nil
      raise ArgumentError.new("Hy-Nomas text refiner plan has been released") if @released
    end

    private def arena : Arena
      @arena || raise ArgumentError.new("Hy-Nomas text refiner Arena is unavailable")
    end

    private def weight_stage : ML::MetalBuffer
      @weight_stage || raise ArgumentError.new("Hy-Nomas text refiner linear stage is unavailable")
    end

    private def checkpoint(name : String) : ML::MetalBuffer
      @checkpoints[name]
    end

    private def allocate_checkpoint(name : String, count : Int32,
                                    checkpoint_name : String = name) : ML::MetalBuffer
      buffer = arena.alloc("hy_nomas_text_refiner.#{name}", count)
      @checkpoints[checkpoint_name] = buffer
      buffer
    end

    private def allocate_buffers! : Nil
      @weight_stage = arena.alloc("hy_nomas_text_refiner.weight_stage",
        (HIDDEN_DIM.to_i64 * FFN_DIM).to_i32)

      allocate_checkpoint("input_compact", @valid_tokens * INPUT_DIM)
      allocate_checkpoint("masked_mean", INPUT_DIM)
      timestep = arena.upload("hy_nomas_text_refiner.timestep", [@timestep])
      # The original scalar is retained as a setup-time GPU input, not read
      # back or recomputed on the host during encode.
      @checkpoints["timestep"] = timestep
      sinusoidal = allocate_checkpoint("timestep_sinusoidal", TIMESTEP_DIM,
        "refiner_timestep_sinusoidal")
      @checkpoints["time_in.sinusoidal"] = sinusoidal
      allocate_checkpoint("refiner_timestep_mlp1", HIDDEN_DIM)
      allocate_checkpoint("refiner_timestep_silu", HIDDEN_DIM)
      allocate_checkpoint("refiner_timestep", HIDDEN_DIM)
      allocate_checkpoint("context_linear1", HIDDEN_DIM)
      allocate_checkpoint("context_silu", HIDDEN_DIM)
      allocate_checkpoint("context_embed", HIDDEN_DIM)
      allocate_checkpoint("c", HIDDEN_DIM)
      allocate_checkpoint("time_in_mlp1", HIDDEN_DIM, "time_in.mlp1")
      allocate_checkpoint("time_in_silu", HIDDEN_DIM, "time_in.silu")
      allocate_checkpoint("vec", HIDDEN_DIM)
      allocate_checkpoint("input_embed", @valid_tokens * HIDDEN_DIM)

      BLOCKS.times do |block|
        prefix = "block#{block}"
        allocate_checkpoint("#{prefix}.condition_silu", HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.adaLN", 2 * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.gate_msa", HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.gate_mlp", HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.norm1", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.qkv", @valid_tokens * 3 * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.q", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.k", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.v", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.attention", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.proj", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.residual1", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.norm2", @valid_tokens * HIDDEN_DIM)
        allocate_checkpoint("#{prefix}.mlp_fc1", @valid_tokens * FFN_DIM)
        allocate_checkpoint("#{prefix}.mlp_silu", @valid_tokens * FFN_DIM)
        allocate_checkpoint("#{prefix}.mlp_fc2", @valid_tokens * HIDDEN_DIM)
        output = allocate_checkpoint("#{prefix}.output", @valid_tokens * HIDDEN_DIM)
        @checkpoints["#{prefix}.residual2"] = output
      end
      @checkpoints["projected_text"] = checkpoint("block1.output")

      @owned_bindings = [] of Tuple(ML::MetalBuffer, Int64)
      arena.buffers.each_value do |buffer|
        @owned_bindings << {buffer, buffer.size}
      end
      @external_weights.each { |_, buffer, _, _| @all_external << buffer unless @all_external.any?(&.same?(buffer)) }
    end

    private def validate_source! : Nil
      expected_bytes = TOKEN_COUNT.to_i64 * INPUT_DIM * WeightDType::BF16.bytes_per_element
      unless @source.valid? && @source.storage_mode == ML::StorageMode::Shared &&
             @source.size == expected_bytes
        raise ArgumentError.new("Hy-Nomas text refiner input must be exact shared BF16 [1,1280,3584]")
      end
      MetalOps.check_storage(@source, TOKEN_COUNT.to_i64 * INPUT_DIM, WeightDType::BF16)
    end

    private def validate_weights! : Nil
      seen = [] of ML::MetalBuffer
      self.class.checkpoint_bindings.each do |binding|
        unless @weights.allocated?(binding.target)
          raise ArgumentError.new("Missing Hy-Nomas text refiner weight #{binding.target}")
        end
        weight = @weights[binding.target]
        dtype = @weights.dtype(binding.target)
        unless dtype.f32? || dtype.bf16?
          raise ArgumentError.new("Hy-Nomas text refiner weights must be F32 or BF16")
        end
        count = binding.shape.reduce(1_i64) { |product, dimension| product * dimension }
        expected_bytes = count * dtype.bytes_per_element
        unless weight.valid? && weight.storage_mode == ML::StorageMode::Shared &&
               weight.size == expected_bytes
          raise ArgumentError.new("Hy-Nomas text refiner weight #{binding.target} has invalid size or storage")
        end
        MetalOps.check_storage(weight, count, dtype)
        if MetalOps.overlap?(@source, weight) || seen.any? { |other| MetalOps.overlap?(other, weight) }
          raise ArgumentError.new("Hy-Nomas text refiner weights and input must not alias")
        end
        seen << weight
        @external_weights << {binding.target, weight, dtype, expected_bytes}
      end
      unless self.class.checkpoint_bindings.size == 42 && @external_weights.size == 42
        raise ArgumentError.new("Hy-Nomas text refiner checkpoint subset must contain exactly 42 tensors")
      end
    end

    private def prepare_pipelines! : Nil
      {COMPACT_KERNEL, MEAN_KERNEL, TIME_KERNEL, SPLIT_KERNEL, ADALN_KERNEL,
       RESIDUAL_KERNEL, AFFINE_KERNEL, ROUND_KERNEL}.each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) do
          ML::Metal::ComputePipeline.new(name, SOURCE)
        end
      end
    end

    private def prepare_plans! : Nil
      MetalLinearMPS.prepare!
      @norm_plan = HyNomasMPSLayerNorm::Plan.new(HIDDEN_DIM, 1e-6_f32)
      prepare_activation("refiner_timestep_silu", "refiner_timestep_mlp1", [HIDDEN_DIM])
      prepare_activation("context_silu", "context_linear1", [HIDDEN_DIM])
      prepare_activation("time_in.silu", "time_in.mlp1", [HIDDEN_DIM])
      BLOCKS.times do |block|
        prefix = "block#{block}"
        prepare_activation("#{prefix}.condition_silu", "c", [HIDDEN_DIM])
        prepare_activation("#{prefix}.mlp_silu", "#{prefix}.mlp_fc1", [@valid_tokens, FFN_DIM])
        @attention_plans << HyNomasAttentionMPSPlan.new(1, @valid_tokens, HEADS,
          HEAD_DIM, checkpoint("#{prefix}.q"), checkpoint("#{prefix}.k"),
          checkpoint("#{prefix}.v"), checkpoint("#{prefix}.attention"))
      end

      prepare_linear("refiner_timestep_mlp1", "refiner_timestep_sinusoidal",
        "refiner_timestep_mlp1", "txt_in.t_embedder.mlp.0", 1, TIMESTEP_DIM, HIDDEN_DIM)
      prepare_linear("refiner_timestep", "refiner_timestep_silu", "refiner_timestep",
        "txt_in.t_embedder.mlp.2", 1, HIDDEN_DIM, HIDDEN_DIM)
      prepare_linear("context_linear1", "masked_mean", "context_linear1",
        "txt_in.c_embedder.linear_1", 1, INPUT_DIM, HIDDEN_DIM)
      prepare_linear("context_embed", "context_silu", "context_embed",
        "txt_in.c_embedder.linear_2", 1, HIDDEN_DIM, HIDDEN_DIM)
      prepare_linear("input_embed", "input_compact", "input_embed",
        "txt_in.input_embedder", @valid_tokens, INPUT_DIM, HIDDEN_DIM)
      prepare_linear("time_in.mlp1", "time_in.sinusoidal", "time_in.mlp1",
        "time_in.mlp.0", 1, TIMESTEP_DIM, HIDDEN_DIM)
      prepare_linear("time_in.mlp2", "time_in.silu", "vec",
        "time_in.mlp.2", 1, HIDDEN_DIM, HIDDEN_DIM)
      BLOCKS.times { |block| prepare_block_linears(block) }
    end

    private def prepare_block_linears(block : Int32) : Nil
      prefix = "block#{block}"
      source_prefix = "txt_in.individual_token_refiner.blocks.#{block}"
      prepare_linear("#{prefix}.adaLN", "#{prefix}.condition_silu", "#{prefix}.adaLN",
        "#{source_prefix}.adaLN_modulation.1", 1, HIDDEN_DIM, 2 * HIDDEN_DIM)
      prepare_linear("#{prefix}.qkv", "#{prefix}.norm1", "#{prefix}.qkv",
        "#{source_prefix}.self_attn_qkv", @valid_tokens, HIDDEN_DIM, 3 * HIDDEN_DIM)
      prepare_linear("#{prefix}.proj", "#{prefix}.attention", "#{prefix}.proj",
        "#{source_prefix}.self_attn_proj", @valid_tokens, HIDDEN_DIM, HIDDEN_DIM)
      prepare_linear("#{prefix}.mlp_fc1", "#{prefix}.norm2", "#{prefix}.mlp_fc1",
        "#{source_prefix}.mlp.fc1", @valid_tokens, HIDDEN_DIM, FFN_DIM)
      prepare_linear("#{prefix}.mlp_fc2", "#{prefix}.mlp_silu", "#{prefix}.mlp_fc2",
        "#{source_prefix}.mlp.fc2", @valid_tokens, FFN_DIM, HIDDEN_DIM)
    end

    private def prepare_activation(name : String, input_name : String,
                                   shape : Array(Int32)) : Nil
      input = checkpoint(input_name)
      output = checkpoint(name)
      plan = HyNomasMPSActivation::SiLUPlan.new(shape, input, output)
      @activations[name] = plan
      @activation_plans << plan
    end

    private def prepare_linear(name : String, input_name : String, output_name : String,
                               weight_prefix : String, rows : Int32,
                               input_dim : Int32, output_dim : Int32) : Nil
      weight_name = "#{weight_prefix}.weight"
      bias_name = "#{weight_prefix}.bias"
      weight = @weights[weight_name]
      bias = @weights[bias_name]
      weight_dtype = @weights.dtype(weight_name)
      bias_dtype = @weights.dtype(bias_name)
      input = checkpoint(input_name)
      output = checkpoint(output_name)
      # WanT5Config's shared plan validator caps dim/heads at 256. The MPS
      # FC2 descriptor does not use heads, so select a divisor that satisfies
      # that bound for each supported projection input width.
      cfg = WanT5Config.new(vocab: 1, dim: input_dim, ffn_dim: output_dim,
        heads: input_dim // 256, layers: 1, batches: 1, tokens: rows)
      plan = WanT5MPSLinearPlan.new(cfg, input_dim, output_dim, input,
        weight_stage, output)
      @linear_plans << plan
      @linears[name] = LinearEntry.new(input, output, weight, bias,
        weight_dtype, bias_dtype, plan, rows, input_dim, output_dim)
    end

    private def encode_compaction(e : ML::Metal::ComputeEncoder) : Nil
      count = @valid_tokens * INPUT_DIM
      e.set_pipeline(@pipelines[COMPACT_KERNEL])
        .set_buffer(@source, 0)
        .set_buffer(checkpoint("input_compact"), 1, ML::Metal::BufferAccess::Write)
        .set_value(@valid_tokens.to_u32, 2)
        .set_value(INPUT_DIM.to_u32, 3)
        .dispatch_1d(count)
    end

    private def encode_mean(e : ML::Metal::ComputeEncoder) : Nil
      e.set_pipeline(@pipelines[MEAN_KERNEL])
        .set_buffer(checkpoint("input_compact"), 0)
        .set_buffer(checkpoint("masked_mean"), 1, ML::Metal::BufferAccess::Write)
        .set_value(@valid_tokens.to_u32, 2)
        .set_value(INPUT_DIM.to_u32, 3)
        .dispatch_1d(INPUT_DIM)
    end

    private def encode_sinusoidal(e : ML::Metal::ComputeEncoder) : Nil
      e.set_pipeline(@pipelines[TIME_KERNEL])
        .set_buffer(checkpoint("timestep"), 0)
        .set_buffer(checkpoint("refiner_timestep_sinusoidal"), 1, ML::Metal::BufferAccess::Write)
        .set_value((TIMESTEP_DIM // 2).to_u32, 2)
        .dispatch_1d(TIMESTEP_DIM // 2)
    end

    private def encode_linear(chain : ML::Metal::MPSCommandChain, name : String) : Nil
      entry = @linears[name]
      MetalLinearMPS.project(chain.encoder, entry.input, entry.weight, entry.bias,
        entry.output, entry.rows, entry.input_dim, entry.output_dim,
        entry.weight_dtype, entry.bias_dtype, entry.plan, weight_stage)
    end

    private def encode_activation(chain : ML::Metal::MPSCommandChain, name : String) : Nil
      @activations[name].encode(chain)
    end

    private def encode_block(chain : ML::Metal::MPSCommandChain, block : Int32) : Nil
      prefix = "block#{block}"
      source = block == 0 ? checkpoint("input_embed") : checkpoint("block0.output")
      encode_activation(chain, "#{prefix}.condition_silu")
      encode_linear(chain, "#{prefix}.adaLN")
      split_adaln(chain.encoder, checkpoint("#{prefix}.adaLN"),
        checkpoint("#{prefix}.gate_msa"), checkpoint("#{prefix}.gate_mlp"))

      norm_affine(chain.encoder, source, checkpoint("#{prefix}.norm1"),
        "txt_in.individual_token_refiner.blocks.#{block}.norm1")
      encode_linear(chain, "#{prefix}.qkv")
      split_qkv(chain.encoder, checkpoint("#{prefix}.qkv"),
        checkpoint("#{prefix}.q"), checkpoint("#{prefix}.k"),
        checkpoint("#{prefix}.v"))
      if @attention_precision.bf16_boundary_control?
        round_bf16(chain.encoder, checkpoint("#{prefix}.q"), @valid_tokens * HIDDEN_DIM)
        round_bf16(chain.encoder, checkpoint("#{prefix}.k"), @valid_tokens * HIDDEN_DIM)
        round_bf16(chain.encoder, checkpoint("#{prefix}.v"), @valid_tokens * HIDDEN_DIM)
      end
      @attention_plans[block].encode(chain)
      if @attention_precision.bf16_boundary_control?
        round_bf16(chain.encoder, checkpoint("#{prefix}.attention"), @valid_tokens * HIDDEN_DIM)
      end
      encode_linear(chain, "#{prefix}.proj")
      gated_residual(chain.encoder, source, checkpoint("#{prefix}.proj"),
        checkpoint("#{prefix}.gate_msa"), checkpoint("#{prefix}.residual1"))

      norm_affine(chain.encoder, checkpoint("#{prefix}.residual1"),
        checkpoint("#{prefix}.norm2"),
        "txt_in.individual_token_refiner.blocks.#{block}.norm2")
      encode_linear(chain, "#{prefix}.mlp_fc1")
      encode_activation(chain, "#{prefix}.mlp_silu")
      encode_linear(chain, "#{prefix}.mlp_fc2")
      gated_residual(chain.encoder, checkpoint("#{prefix}.residual1"),
        checkpoint("#{prefix}.mlp_fc2"), checkpoint("#{prefix}.gate_mlp"),
        checkpoint("#{prefix}.output"))
    end

    private def gated_residual(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                               delta : ML::MetalBuffer, gate : ML::MetalBuffer,
                               output : ML::MetalBuffer) : Nil
      e.set_pipeline(@pipelines[RESIDUAL_KERNEL])
        .set_buffer(input, 0).set_buffer(delta, 1).set_buffer(gate, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value((@valid_tokens * HIDDEN_DIM).to_u32, 4)
        .set_value(HIDDEN_DIM.to_u32, 5).dispatch_1d(@valid_tokens * HIDDEN_DIM)
    end

    private def split_qkv(e : ML::Metal::ComputeEncoder, qkv : ML::MetalBuffer,
                          q : ML::MetalBuffer, k : ML::MetalBuffer,
                          v : ML::MetalBuffer) : Nil
      e.set_pipeline(@pipelines[SPLIT_KERNEL])
        .set_buffer(qkv, 0).set_buffer(q, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(k, 2, ML::Metal::BufferAccess::Write)
        .set_buffer(v, 3, ML::Metal::BufferAccess::Write)
        .set_value(@valid_tokens.to_u32, 4).set_value(HIDDEN_DIM.to_u32, 5)
        .dispatch_1d(@valid_tokens * HIDDEN_DIM)
    end

    private def split_adaln(e : ML::Metal::ComputeEncoder, adaln : ML::MetalBuffer,
                            gate_msa : ML::MetalBuffer, gate_mlp : ML::MetalBuffer) : Nil
      e.set_pipeline(@pipelines[ADALN_KERNEL])
        .set_buffer(adaln, 0).set_buffer(gate_msa, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(gate_mlp, 2, ML::Metal::BufferAccess::Write)
        .set_value(HIDDEN_DIM.to_u32, 3).dispatch_1d(HIDDEN_DIM)
    end

    private def norm_affine(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                            output : ML::MetalBuffer, prefix : String) : Nil
      @norm_plan.not_nil!.encode(e, input, output, @valid_tokens)
      weight_name = "#{prefix}.weight"
      bias_name = "#{prefix}.bias"
      weight = @weights[weight_name]
      bias = @weights[bias_name]
      weight_dtype = @weights.dtype(weight_name)
      bias_dtype = @weights.dtype(bias_name)
      count = @valid_tokens * HIDDEN_DIM
      MetalOps.distinct(output, weight, bias)
      e.set_pipeline(@pipelines[AFFINE_KERNEL])
        .set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(weight, 1).set_buffer(bias, 2)
        .set_buffer(weight, 3).set_buffer(bias, 4)
        .set_value(count.to_u32, 5).set_value(HIDDEN_DIM.to_u32, 6)
        .set_value(weight_dtype.bf16? ? 1_u32 : 0_u32, 7)
        .set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 8)
        .dispatch_1d(count)
    end

    private def round_bf16(e : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer,
                           count : Int32) : Nil
      e.set_pipeline(@pipelines[ROUND_KERNEL])
        .set_buffer(buffer, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_value(count.to_u32, 1).dispatch_1d(count)
    end

    private def validate_bindings! : Nil
      validate_source!
      unless @mask.size == TOKEN_COUNT && self.class.valid_tokens(@mask) == @valid_tokens
        raise ArgumentError.new("Hy-Nomas text refiner immutable mask binding changed")
      end
      @owned_bindings.each do |buffer, expected_bytes|
        unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared &&
               buffer.size == expected_bytes
          raise ArgumentError.new("Hy-Nomas text refiner scratch binding changed")
        end
      end
      @external_weights.each do |name, buffer, dtype, expected_bytes|
        current = @weights.buffers[name]?
        unless current && current.same?(buffer) && buffer.valid? &&
               @weights.dtype(name) == dtype && buffer.storage_mode == ML::StorageMode::Shared &&
               buffer.size == expected_bytes
          raise ArgumentError.new("Hy-Nomas text refiner weight binding changed: #{name}")
        end
      end
      @all_external.each do |external|
        @owned_bindings.each do |owned, _|
          raise ArgumentError.new("Hy-Nomas text refiner buffers must not alias") if MetalOps.overlap?(external, owned)
        end
      end
      @owned_bindings.each_with_index do |(left, _), index|
        @owned_bindings.each_with_index do |(right, _), other_index|
          next if other_index <= index
          raise ArgumentError.new("Hy-Nomas text refiner owned buffers must not alias") if MetalOps.overlap?(left, right)
        end
      end
      @linears.each_value do |entry|
        entry.plan.validate_buffers!(entry.input, weight_stage, entry.output)
        MetalOps.distinct(entry.output, entry.input, entry.weight, entry.bias)
      end
      BLOCKS.times do |block|
        prefix = "block#{block}"
        @attention_plans[block].validate_bindings!(checkpoint("#{prefix}.q"),
          checkpoint("#{prefix}.k"), checkpoint("#{prefix}.v"),
          checkpoint("#{prefix}.attention"))
      end
    end

    private def release_prepared_plans : Nil
      @activation_plans.reverse_each(&.release)
      @activation_plans.clear
      @activations.clear
      @attention_plans.reverse_each(&.release)
      @attention_plans.clear
      @linear_plans.reverse_each(&.release)
      @linear_plans.clear
      @norm_plan = nil
    end

    private def retain_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_plans << self unless @@live_plans.any? { |current| current.same?(self) }
      end
    end

    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_plans.reject! { |current| current.same?(self) }
      end
    end
  end
end
