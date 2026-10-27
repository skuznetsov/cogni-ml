# Opt-in MPS execution for one already-constructed Hy-Nomas dual-stream block.
# The base block deliberately does not require this file, keeping its native
# encoder path free of MPS bridge dependencies.
require "./hy_nomas_block"
require "./metal_linear_mps"
require "./hy_nomas_attention_mps"
require "./hy_nomas_attention_compact"
require "./hy_nomas_mps_norm"
require "./hy_nomas_mps_pointwise"
require "./hy_nomas_mps_activation"
require "./hy_nomas_mps_rms"

module ML::Video
  # Fixed-buffer MPS plans for every linear operation plus one canonical or
  # opt-in compact SDPA plan. The caller owns this workspace and its input,
  # weight, and block scratch Arenas; the weight expansion and attention stages are
  # owned here. Reuse is serial: keep all bindings alive and immutable through
  # the command chain's terminal wait.
  class HyNomasMPSWorkspace
    WEIGHT_STAGE_KEY       = "hy_nomas_mps.weight_stage"
    ATTENTION_CLEAR_KERNEL = "hy_nomas_clear_joint_attention_f32"
    ATTENTION_CLEAR_SOURCE = MetalOps::SOURCE + <<-METAL
      kernel void hy_nomas_clear_joint_attention_f32(
          device float *output [[buffer(0)]],
          constant uint &count [[buffer(1)]],
          uint index [[thread_position_in_grid]]) {
        if (index < count) output[index] = 0.0f;
      }
      METAL

    private record SerialBinding, block : HyNomasDualStreamBlock,
      img : ML::MetalBuffer, txt : ML::MetalBuffer,
      scratch : Array(ML::MetalBuffer), weights : Array(ML::MetalBuffer),
      linear : Hash(String, HyNomasDualStreamBlock::MPSLinearOperation),
      rms : Hash(String, HyNomasDualStreamBlock::MPSRMSOperation)

    # MPSCommandChain has an explicit terminal wait but no deferred-resource
    # retention hook. Root each prepared workspace until its owner explicitly
    # releases it after that wait, so plan finalizers cannot invalidate graph
    # state or staging buffers while encoded work is still in flight. If a
    # caller abandons a workspace, it remains rooted until process exit; this
    # deliberately prefers a leak over releasing storage before GPU completion.
    @@lifetime_mutex = Mutex.new
    @@live_workspaces = [] of HyNomasMPSWorkspace

    @block : HyNomasDualStreamBlock
    @config : HyNomasConfig
    @attention_tokens : Int32
    @arena : Arena
    @weight_stage : ML::MetalBuffer? = nil
    @operations = {} of String => HyNomasDualStreamBlock::MPSLinearOperation
    @plans = {} of String => WanT5MPSLinearPlan
    @scratch_buffers : Array(ML::MetalBuffer)
    @weight_buffers : Array(ML::MetalBuffer)
    @attention_plan : (HyNomasAttentionMPSPlan | HyNomasCompactAttentionMPSPlan)? = nil
    @norm_plan : HyNomasMPSLayerNorm::Plan
    @allow_shared_modulation : Bool
    @attention_clear_pipeline : ML::Metal::ComputePipeline? = nil
    @pointwise_plan : HyNomasMPSPointwise::Plan? = nil
    @activation_operations = {} of String => HyNomasDualStreamBlock::MPSActivationOperation
    @activation_plans = {} of String => HyNomasMPSActivation::UnaryPlan
    @rms_operations = {} of String => HyNomasDualStreamBlock::MPSRMSOperation
    @rms_plans = {} of String => HyNomasMPSRMS::Plan
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @serial_bindings = [] of SerialBinding
    @serial_buffers = [] of ML::MetalBuffer

    getter config : HyNomasConfig
    getter attention_tokens : Int32

    # Resolve the supported contiguous text-prefix attention geometry without
    # allocating native resources. Full-sequence behavior remains the default.
    def self.attention_tokens_for(config : HyNomasConfig,
                                  valid_text_tokens : Int32?) : Int32
      config.validate!
      return config.joint_tokens unless valid = valid_text_tokens
      unless config.batches == 1
        raise ArgumentError.new("Hy-Nomas text-prefix attention requires batch-one geometry")
      end
      unless valid > 0 && valid <= config.text_tokens
        raise ArgumentError.new("Hy-Nomas valid text prefix must be positive and no greater than text_tokens")
      end
      config.video_tokens + valid
    end

    private def weight_stage : ML::MetalBuffer
      @weight_stage || raise ArgumentError.new("Hy-Nomas MPS weight stage is unavailable")
    end

    def initialize(@block : HyNomasDualStreamBlock,
                   img : ML::MetalBuffer, txt : ML::MetalBuffer,
                   vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                   video_rope_cos : ML::MetalBuffer,
                   video_rope_sin : ML::MetalBuffer,
                   *, attention_query_chunk : Int32 = 0,
                   attention_heads_per_group : Int32 = 0,
                   allow_shared_modulation : Bool = false,
                   valid_text_tokens : Int32? = nil,
                   text_rms_tokens : Int32? = nil)
      @config = @block.config
      @allow_shared_modulation = allow_shared_modulation
      @attention_tokens = self.class.attention_tokens_for(@config, valid_text_tokens)
      if reduction_tokens = text_rms_tokens
        unless reduction_tokens >= @config.text_tokens
          raise ArgumentError.new("Hy-Nomas text RMS reduction tokens must cover the block text storage")
        end
        HyNomasDualStreamBlock.checked_count(
          @config.batches, reduction_tokens, @config.heads, @config.head_dim,
        )
      end
      if attention_query_chunk != 0 || attention_heads_per_group != 0
        HyNomasCompactAttentionMPSPlan.workspace_geometry(@config.batches,
          @attention_tokens, @config.heads, @config.head_dim,
          attention_query_chunk, attention_heads_per_group)
      end
      @norm_plan = HyNomasMPSLayerNorm::Plan.new(@config.dim, @config.norm_eps)
      @block.mps_validate_inputs!(img, txt, vec, vec_txt, video_rope_cos, video_rope_sin)
      @block.mps_validate_resources!
      @fixed_inputs = [img, txt, vec, vec_txt, video_rope_cos, video_rope_sin]
      @scratch_buffers, @weight_buffers = @block.mps_workspace_buffers

      linear_operations = @block.mps_linear_operations
      unless linear_operations.size == 14
        raise ArgumentError.new("Hy-Nomas MPS workspace requires all 14 block linear operations")
      end
      linear_operations.each do |operation|
        raise ArgumentError.new("Duplicate Hy-Nomas MPS linear operation #{operation.name}") if @operations.has_key?(operation.name)
        @operations[operation.name] = operation
      end

      @arena = Arena.new
      begin
        stage_count = self.class.required_weight_stage_count(@config)
        @weight_stage = @arena.alloc(WEIGHT_STAGE_KEY, stage_count)
        validate_setup_aliases!

        MetalLinearMPS.prepare!
        @pointwise_plan = HyNomasMPSPointwise::Plan.new(@config.dim)
        prepare_linear_plans!(linear_operations)
        @block.mps_activation_operations(vec, vec_txt).each do |operation|
          @activation_operations[operation.name] = operation
          if operation.name.ends_with?(".silu")
            @activation_plans[operation.name] = HyNomasMPSActivation::SiLUPlan.new(
              operation.shape, operation.input, operation.output)
          else
            @activation_plans[operation.name] = HyNomasMPSActivation::GELUPlan.new(
              operation.shape, operation.input, operation.output)
          end
        end
        @block.mps_rms_operations.each do |operation|
          @rms_operations[operation.name] = operation
          reduction_tokens = operation.name.starts_with?("txt.") ? text_rms_tokens : nil
          @rms_plans[operation.name] = HyNomasMPSRMS::Plan.new(@config,
            operation.input, operation.weight.weight, operation.output,
            operation.weight.dtype, operation.tokens,
            reduction_tokens: reduction_tokens)
        end
        if @attention_tokens < @config.joint_tokens
          @attention_clear_pipeline = ML::Metal::PipelineCache.get(ATTENTION_CLEAR_KERNEL) do
            ML::Metal::ComputePipeline.new(ATTENTION_CLEAR_KERNEL, ATTENTION_CLEAR_SOURCE)
          end
        end
        @attention_plan = if attention_query_chunk > 0
                            HyNomasCompactAttentionMPSPlan.new(@config.batches,
                              @attention_tokens, @config.heads, @config.head_dim,
                              @block.checkpoints["joint.query"], @block.checkpoints["joint.key"],
                              @block.checkpoints["joint.value"], @block.checkpoints["joint.attention"],
                              attention_query_chunk, attention_heads_per_group)
                          else
                            HyNomasAttentionMPSPlan.new(@config.batches,
                              @attention_tokens, @config.heads, @config.head_dim,
                              @block.checkpoints["joint.query"], @block.checkpoints["joint.key"],
                              @block.checkpoints["joint.value"], @block.checkpoints["joint.attention"])
                          end
        retain_lifetime_root!
      rescue ex
        release_lifetime_root!
        release_prepared_plans
        @arena.release
        @released = true
        raise ex
      end
    end

    # Deterministic lifecycle introspection for callers/tests that verify the
    # explicit-release contract without relying on garbage-collection timing.
    def self.retained_workspace_count : Int32
      @@lifetime_mutex.synchronize do
        @@live_workspaces.size.to_i32
      end
    end

    # Keep this workspace strongly reachable because MPSGraph/linear plan
    # finalizers may otherwise release resources before the caller's terminal
    # wait. Registration occurs once at setup, never in the encode path.
    private def retain_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        unless @@live_workspaces.any? { |current| current.same?(self) }
          @@live_workspaces << self
        end
      end
    end

    # Explicit release is permitted only after the last chain completed; drop
    # the strong root only after every owned plan and buffer was released.
    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_workspaces.reject! { |current| current.same?(self) }
      end
    end

    def linear_plan_count : Int32
      @plans.size.to_i32
    end

    def weight_stage_elements : Int32
      weight_stage.size.to_i32 // sizeof(Float32)
    end

    def resident_bytes : Int64
      @arena.resident_bytes + (@attention_plan.try(&.resident_bytes) || 0_i64) + @rms_plans.values.sum(&.resident_bytes)
    end

    # Setup-only admission for a bounded serial owner. All graph-bound
    # intermediates remain fixed; each block has distinct final output banks.
    # No plans or GPU storage are created when selecting a registered block.
    def register_serial_blocks!(blocks : Array(HyNomasDualStreamBlock),
                                inputs : Array(Tuple(ML::MetalBuffer, ML::MetalBuffer)),
                                *, full_dit : Bool = false) : Nil
      raise ArgumentError.new("Hy-Nomas serial registration must precede encode") if @released || @last_chain || !@serial_bindings.empty?
      count_valid = full_dit ? blocks.size == 54 : (2..8).includes?(blocks.size)
      unless count_valid && inputs.size == blocks.size && blocks[0].same?(@block)
        expected = full_dit ? "exactly 54" : "two to eight"
        raise ArgumentError.new("Hy-Nomas serial registration requires #{expected} ordered blocks beginning with the prepared block")
      end
      bindings = [] of SerialBinding
      allowed = @fixed_inputs.dup
      plan = HyNomasDualStreamBlock.scratch_plan(@config)
      output_indices = plan.each_index.select { |index| {"hy_nomas.img.output", "hy_nomas.txt.output"}.includes?(plan[index][0]) }.to_a
      raise ArgumentError.new("Hy-Nomas serial output schema changed") unless output_indices.size == 2
      output_banks = output_indices.map { |index| @scratch_buffers[index] }
      output_banks.each_with_index do |buffer, index|
        output_banks[0...index].each { |other| MetalOps.distinct(buffer, other) }
        @fixed_inputs.each { |other| MetalOps.distinct(buffer, other) }
      end
      blocks.each_with_index do |block, index|
        if blocks.each_with_index.any? { |candidate, candidate_index| candidate_index < index && candidate.same?(block) }
          raise ArgumentError.new("Hy-Nomas serial registration rejects duplicate block instances")
        end
        raise ArgumentError.new("Hy-Nomas serial block configuration changed") unless block.config == @config
        img, txt = inputs[index]
        expected_img = index == 0 ? @fixed_inputs[0] : blocks[index - 1].image_output
        expected_txt = index == 0 ? @fixed_inputs[1] : blocks[index - 1].text_output
        unless img.same?(expected_img) && txt.same?(expected_txt)
          raise ArgumentError.new("Hy-Nomas serial input banks must follow the registered predecessor")
        end
        block.mps_validate_inputs!(img, txt, @fixed_inputs[2], @fixed_inputs[3], @fixed_inputs[4], @fixed_inputs[5])
        block.mps_validate_resources!
        scratch, weights = block.mps_workspace_buffers
        scratch.each_with_index do |buffer, slot|
          if output_indices.includes?(slot)
            if index == 0
              unless buffer.same?(@scratch_buffers[slot])
                raise ArgumentError.new("Hy-Nomas serial first-block output bank changed")
              end
            else
              output_banks.each { |other| MetalOps.distinct(buffer, other) }
              @scratch_buffers.each_with_index do |other, other_slot|
                next if output_indices.includes?(other_slot)
                MetalOps.distinct(buffer, other)
              end
              @fixed_inputs.each { |other| MetalOps.distinct(buffer, other) }
              output_banks << buffer
            end
          elsif !buffer.same?(@scratch_buffers[slot])
            raise ArgumentError.new("Hy-Nomas serial intermediate buffer identity changed: #{plan[slot][0]}")
          end
        end
        linear = block.mps_linear_operations.to_h { |operation| {operation.name, operation} }
        rms = block.mps_rms_operations.to_h { |operation| {operation.name, operation} }
        raise ArgumentError.new("Hy-Nomas serial operation schema changed") unless linear.keys.sort == @operations.keys.sort && rms.keys.sort == @rms_operations.keys.sort
        linear.each do |name, operation|
          original = @operations[name]
          unless operation.input.same?(original.input) && operation.output.same?(original.output) &&
                 operation.rows == original.rows && operation.input_dim == original.input_dim && operation.output_dim == original.output_dim
            raise ArgumentError.new("Hy-Nomas serial linear graph binding changed: #{name}")
          end
          validate_linear_operation!(operation)
        end
        rms.each do |name, operation|
          original = @rms_operations[name]
          unless operation.input.same?(original.input) && operation.output.same?(original.output) && operation.tokens == original.tokens
            raise ArgumentError.new("Hy-Nomas serial RMS graph binding changed: #{name}")
          end
        end
        activations = block.mps_activation_operations(@fixed_inputs[2], @fixed_inputs[3])
        unless activations.size == @activation_operations.size && activations.all? { |operation| operation == @activation_operations[operation.name]? }
          raise ArgumentError.new("Hy-Nomas serial activation graph binding changed")
        end
        allowed.concat(scratch).concat([img, txt])
        bindings << SerialBinding.new(block, img, txt, scratch, weights, linear, rms)
      end
      unless output_banks.size == blocks.size * output_indices.size
        raise ArgumentError.new("Hy-Nomas serial output bank inventory changed")
      end
      # Cross-block aliases are not covered by each block's local validator.
      bindings.each do |binding|
        binding.weights.each do |weight|
          allowed.each { |buffer| MetalOps.distinct(weight, buffer) }
          MetalOps.distinct(weight, weight_stage)
        end
      end
      @serial_bindings = bindings
      @serial_buffers = allowed
    end

    # Verify plan and buffer identity, geometry, aliases, and chain lifecycle
    # without allocating or submitting work. The serial owner's all-block
    # preflight uses bind_chain: false so a rejected successor does not bind an
    # otherwise unused command chain; block dispatch uses the default binding.
    def validate_for!(block : HyNomasDualStreamBlock,
                      chain : ML::Metal::MPSCommandChain,
                      img : ML::MetalBuffer, txt : ML::MetalBuffer,
                      vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                      video_rope_cos : ML::MetalBuffer,
                      video_rope_sin : ML::MetalBuffer,
                      *, bind_chain : Bool = true) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      # Check the common lease before changing active block metadata. Sibling
      # registered blocks must never overwrite shared stages on a live chain.
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas MPS workspace cannot span in-flight command chains")
        end
      end
      serial = @serial_bindings.find { |binding| binding.block.same?(block) }
      unless (@serial_bindings.empty? ? @block.same?(block) : !serial.nil?) && @config == block.config
        raise ArgumentError.new("Hy-Nomas MPS workspace belongs to a different block")
      end
      block.mps_validate_inputs!(img, txt, vec, vec_txt, video_rope_cos, video_rope_sin)
      block.mps_validate_resources!
      input_matches = if binding = serial
                        binding.img.same?(img) && binding.txt.same?(txt) &&
                          @fixed_inputs[2].same?(vec) && @fixed_inputs[3].same?(vec_txt) &&
                          @fixed_inputs[4].same?(video_rope_cos) && @fixed_inputs[5].same?(video_rope_sin)
                      else
                        fixed_input?(img, txt, vec, vec_txt, video_rope_cos, video_rope_sin)
                      end
      unless input_matches
        raise ArgumentError.new("Hy-Nomas MPS workspace input identity changed")
      end
      scratch = serial.try(&.scratch) || @scratch_buffers
      weights = serial.try(&.weights) || @weight_buffers
      unless block.mps_workspace_resources_match?(scratch, weights)
        raise ArgumentError.new("Hy-Nomas MPS workspace block buffer identity changed")
      end
      validate_workspace_allocations!
      validate_linear_plans!(serial.try(&.linear) || @operations)
      validate_attention_plan!(block)

      # Validate the chain before any native linear work is appended. The
      # attention plan cannot move between in-flight chains.
      return unless bind_chain
      if binding = serial
        @operations = binding.linear
        @rms_operations = binding.rms
      end
      @last_chain = chain
    end

    # Encode the single full-sequence unmasked attention operation. The plan
    # pauses/resumes the chain around MPSGraph; the caller obtains its refreshed
    # encoder immediately afterward.
    def encode_attention(chain : ML::Metal::MPSCommandChain,
                         query : ML::MetalBuffer, key : ML::MetalBuffer,
                         value : ML::MetalBuffer, output : ML::MetalBuffer,
                         config : HyNomasConfig) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      unless config == @config &&
             query.same?(@block.checkpoints["joint.query"]) &&
             key.same?(@block.checkpoints["joint.key"]) &&
             value.same?(@block.checkpoints["joint.value"]) &&
             output.same?(@block.checkpoints["joint.attention"])
        raise ArgumentError.new("Hy-Nomas MPS attention binding changed")
      end
      plan = @attention_plan || raise ArgumentError.new("Hy-Nomas MPS attention plan is unavailable")
      if @attention_tokens < @config.joint_tokens
        count = HyNomasDualStreamBlock.checked_count(@config.batches,
          @config.joint_tokens, @config.dim)
        pipeline = @attention_clear_pipeline || raise ArgumentError.new("Hy-Nomas attention clear pipeline is unavailable")
        chain.encoder.set_pipeline(pipeline)
          .set_buffer(output, 0, ML::Metal::BufferAccess::Write)
          .set_value(count.to_u32, 1)
          .dispatch_1d(count)
      end
      plan.encode(chain)
    end

    # Backend hook: derive the encoder from the chain registered by block
    # preflight so direct callers cannot submit staging work on an untracked
    # command and then release its storage early.
    def project(chain : ML::Metal::MPSCommandChain, name : String,
                input : ML::MetalBuffer,
                weights : HyNomasDualStreamBlock::LinearWeights,
                output : ML::MetalBuffer, rows : Int32,
                input_dim : Int32, output_dim : Int32) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      unless @last_chain && @last_chain.not_nil!.same?(chain)
        raise ArgumentError.new("Hy-Nomas MPS projection requires block preflight on this chain")
      end
      e = chain.encoder
      operation = @operations[name]?
      raise ArgumentError.new("Unknown Hy-Nomas MPS linear operation #{name}") unless operation
      plan = @plans[name]?
      raise ArgumentError.new("Missing Hy-Nomas MPS linear plan #{name}") unless plan
      unless operation.input.same?(input) && operation.output.same?(output) &&
             operation.weights.weight.same?(weights.weight) && operation.weights.bias.same?(weights.bias) &&
             operation.weights.weight_dtype == weights.weight_dtype &&
             operation.weights.bias_dtype == weights.bias_dtype &&
             operation.rows == rows && operation.input_dim == input_dim && operation.output_dim == output_dim
        raise ArgumentError.new("Hy-Nomas MPS linear binding changed for #{name}")
      end
      plan.validate_buffers!(input, weight_stage, output)
      MetalLinearMPS.project(e, input, weights.weight, weights.bias, output,
        rows, input_dim, output_dim, weights.weight_dtype, weights.bias_dtype,
        plan, weight_stage, apply_bias: true)
    end

    def layer_norm(e : ML::Metal::ComputeEncoder,
                   input : ML::MetalBuffer, output : ML::MetalBuffer,
                   rows : Int32, dim : Int32, epsilon : Float32) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      unless dim == @config.dim && epsilon == @config.norm_eps
        raise ArgumentError.new("Hy-Nomas MPS LayerNorm shape or epsilon changed")
      end
      @norm_plan.encode(e, input, output, rows)
    end

    def activation(chain : ML::Metal::MPSCommandChain, name : String,
                   input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      operation = @activation_operations[name]?
      unless operation && operation.input.same?(input) && operation.output.same?(output)
        raise ArgumentError.new("Hy-Nomas MPS activation binding changed: #{name}")
      end
      plan = @activation_plans[name]? || raise ArgumentError.new("Missing Hy-Nomas MPS activation plan: #{name}")
      plan.encode(chain)
    end

    def rms_norm(chain : ML::Metal::MPSCommandChain, name : String,
                 input : ML::MetalBuffer, weight : HyNomasDualStreamBlock::NormWeight,
                 output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      operation = @rms_operations[name]?
      unless operation && operation.input.same?(input) && operation.output.same?(output) &&
             operation.weight.weight.same?(weight.weight) && operation.weight.dtype == weight.dtype
        raise ArgumentError.new("Hy-Nomas MPS RMS binding changed: #{name}")
      end
      plan = @rms_plans[name]? || raise ArgumentError.new("Missing Hy-Nomas MPS RMS plan: #{name}")
      plan.encode(chain, weight.weight, weight.dtype)
    end

    def rope(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
             output : ML::MetalBuffer, cosine : ML::MetalBuffer,
             sine : ML::MetalBuffer, tokens : Int32) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      raise ArgumentError.new("Hy-Nomas MPS RoPE geometry changed") unless tokens == @config.video_tokens
      validate_pointwise_bindings!(input, output, cosine, sine)
      plan = @pointwise_plan || raise ArgumentError.new("Hy-Nomas MPS pointwise plan is unavailable")
      plan.rope(e, input, cosine, sine, output, @config.batches,
        tokens, @config.heads, @config.head_dim)
    end

    def modulate(e : ML::Metal::ComputeEncoder,
                 input : ML::MetalBuffer, shift : ML::MetalBuffer,
                 scale : ML::MetalBuffer, output : ML::MetalBuffer,
                 rows : Int32, dim : Int32) : Nil
      validate_pointwise_geometry!(rows, dim)
      validate_pointwise_bindings!(input, shift, scale, output)
      plan = @pointwise_plan || raise ArgumentError.new("Hy-Nomas MPS pointwise plan is unavailable")
      plan.modulate(e, input, shift, scale, output, rows, dim)
    end

    def gated_residual(e : ML::Metal::ComputeEncoder,
                       input : ML::MetalBuffer, delta : ML::MetalBuffer,
                       gate : ML::MetalBuffer, output : ML::MetalBuffer,
                       rows : Int32, dim : Int32) : Nil
      validate_pointwise_geometry!(rows, dim)
      validate_pointwise_bindings!(input, delta, gate, output)
      plan = @pointwise_plan || raise ArgumentError.new("Hy-Nomas MPS pointwise plan is unavailable")
      plan.gated_residual(e, input, delta, gate, output, rows, dim)
    end

    # Release only after the terminal wait for the last chain that used this
    # workspace. Caller-owned weights, inputs, and block scratch remain owned by
    # their original Arenas.
    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas MPS command chain before releasing workspace") unless chain.completed?
      end
      release_prepared_plans
      @arena.release
      @released = true
      release_lifetime_root!
    end

    def release_after_terminal_wait! : Nil
      release
    end

    def self.required_weight_stage_count(config : HyNomasConfig) : Int32
      config.validate!
      dim = config.dim.to_i64
      count = Math.max(6_i64 * dim * dim, dim * config.ffn_dim)
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas MPS weight stage exceeds Int32 storage")
      end
      count.to_i32
    end

    private def fixed_input?(img : ML::MetalBuffer, txt : ML::MetalBuffer,
                             vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                             video_rope_cos : ML::MetalBuffer,
                             video_rope_sin : ML::MetalBuffer) : Bool
      @fixed_inputs[0].same?(img) && @fixed_inputs[1].same?(txt) &&
        @fixed_inputs[2].same?(vec) && @fixed_inputs[3].same?(vec_txt) &&
        @fixed_inputs[4].same?(video_rope_cos) && @fixed_inputs[5].same?(video_rope_sin)
    end

    private def validate_pointwise_geometry!(rows : Int32, dim : Int32) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      unless dim == @config.dim && (rows == @config.image_rows || rows == @config.text_rows)
        raise ArgumentError.new("Hy-Nomas MPS pointwise geometry does not match the block configuration")
      end
    end

    private def validate_pointwise_bindings!(*buffers : ML::MetalBuffer) : Nil
      buffers.each do |buffer|
        unless @scratch_buffers.any? { |scratch| scratch.same?(buffer) } ||
               @fixed_inputs.any? { |input| input.same?(buffer) } ||
               @serial_buffers.any? { |allowed| allowed.same?(buffer) }
          raise ArgumentError.new("Hy-Nomas MPS pointwise binding is outside the prepared block workspace")
        end
      end
    end

    private def prepare_linear_plans!(operations : Array(HyNomasDualStreamBlock::MPSLinearOperation)) : Nil
      operations.each do |operation|
        unless operation.rows % @config.batches == 0
          raise ArgumentError.new("Hy-Nomas MPS projection rows do not preserve batch geometry: #{operation.name}")
        end
        tokens = operation.rows // @config.batches
        validate_linear_operation!(operation)
        # The prepared plan receives the true projection dimensions below.
        # Keep WanT5Config's unrelated transformer-shape validator neutral so
        # it does not reject valid rectangular Hy-Nomas FC2 projections.
        plan_config = WanT5Config.new(vocab: 1, dim: 1,
          ffn_dim: 1, heads: 1,
          layers: 1, batches: @config.batches, tokens: tokens)
        plan_config.validate!
        plan = WanT5MPSLinearPlan.new(plan_config, operation.input_dim,
          operation.output_dim, operation.input, weight_stage, operation.output)
        @plans[operation.name] = plan
      end
    end

    private def validate_linear_operation!(operation : HyNomasDualStreamBlock::MPSLinearOperation) : Nil
      MetalOps.dims(operation.rows, operation.input_dim, operation.output_dim)
      MetalOps.check(operation.input, operation.rows.to_i64 * operation.input_dim)
      MetalOps.check(operation.output, operation.rows.to_i64 * operation.output_dim)
      MetalOps.check_storage(operation.weights.weight,
        operation.input_dim.to_i64 * operation.output_dim, operation.weights.weight_dtype)
      MetalOps.check_storage(operation.weights.bias, operation.output_dim,
        operation.weights.bias_dtype)
      MetalOps.distinct(operation.output, operation.input, operation.weights.weight, operation.weights.bias)
      MetalOps.distinct(weight_stage, operation.input, operation.weights.weight,
        operation.weights.bias, operation.output)
      unless {operation.input, operation.output, operation.weights.weight,
              operation.weights.bias, weight_stage}.all? { |buffer| buffer.storage_mode == ML::StorageMode::Shared }
        raise ArgumentError.new("Hy-Nomas MPS projections require shared Metal buffers")
      end
    end

    private def validate_setup_aliases! : Nil
      unless @arena[WEIGHT_STAGE_KEY].same?(weight_stage) && @arena.dtype(WEIGHT_STAGE_KEY).f32?
        raise ArgumentError.new("Hy-Nomas MPS workspace Arena bindings changed during setup")
      end
      MetalOps.check(weight_stage, self.class.required_weight_stage_count(@config))
      @fixed_inputs.each do |input|
        MetalOps.distinct(weight_stage, input)
        @weight_buffers.each { |weight| MetalOps.distinct(weight, input) }
      end
      @fixed_inputs.each_with_index do |left, index|
        ((index + 1)...@fixed_inputs.size).each do |other_index|
          if @allow_shared_modulation && index == 2 && other_index == 3 &&
             left.same?(@fixed_inputs[other_index])
            next
          end
          if MetalOps.overlap?(left, @fixed_inputs[other_index])
            raise ArgumentError.new("Hy-Nomas fixed input buffers must not alias")
          end
        end
      end
      @scratch_buffers.each do |buffer|
        MetalOps.distinct(weight_stage, buffer)
      end
      @weight_buffers.each do |buffer|
        MetalOps.distinct(weight_stage, buffer)
      end
    end

    private def validate_workspace_allocations! : Nil
      unless @arena.allocated?(WEIGHT_STAGE_KEY) && @arena[WEIGHT_STAGE_KEY].same?(weight_stage) &&
             @arena.dtype(WEIGHT_STAGE_KEY).f32?
        raise ArgumentError.new("Hy-Nomas MPS workspace Arena binding changed")
      end
      unless weight_stage.valid? && weight_stage.size == self.class.required_weight_stage_count(@config).to_i64 * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas MPS workspace allocation extent changed")
      end
      validate_setup_aliases!
    end

    private def validate_linear_plans!(operations = @operations) : Nil
      raise ArgumentError.new("Hy-Nomas MPS linear plan set is incomplete") unless @plans.size == 14
      operations.each do |name, operation|
        plan = @plans[name]?
        raise ArgumentError.new("Missing Hy-Nomas MPS linear plan #{name}") unless plan
        unless operation.rows % @config.batches == 0 &&
               plan.batches == @config.batches &&
               plan.tokens == operation.rows // @config.batches &&
               plan.input_dim == operation.input_dim && plan.output_dim == operation.output_dim
          raise ArgumentError.new("Hy-Nomas MPS linear plan shape changed for #{name}")
        end
        validate_linear_operation!(operation)
        plan.validate_buffers!(operation.input, weight_stage, operation.output)
      end
    end

    private def validate_attention_plan!(block : HyNomasDualStreamBlock) : Nil
      plan = @attention_plan || raise ArgumentError.new("Hy-Nomas MPS attention plan is unavailable")
      checkpoints = block.checkpoints
      q = checkpoints["joint.query"]
      k = checkpoints["joint.key"]
      v = checkpoints["joint.value"]
      output = checkpoints["joint.attention"]
      unless plan.batches == @config.batches && plan.tokens == @attention_tokens &&
             plan.heads == @config.heads && plan.head_dim == @config.head_dim
        raise ArgumentError.new("Hy-Nomas MPS attention plan bindings changed")
      end
      plan.validate_bindings!(q, k, v, output)
    end

    private def release_prepared_plans : Nil
      @activation_plans.each_value(&.release)
      @activation_plans.clear
      @rms_plans.each_value(&.release)
      @rms_plans.clear
      @attention_plan.try(&.release)
      @attention_plan = nil
      @plans.each_value(&.release)
      @plans.clear
    end
  end

  private class HyNomasMPSBackend < HyNomasDualStreamBlock::EncodeBackend
    @chain : ML::Metal::MPSCommandChain
    @workspace : HyNomasMPSWorkspace

    def initialize(@chain, @workspace, @phase_observer : Proc(String, Nil)? = nil)
    end

    def phase_marker(name : String) : Nil
      @phase_observer.try(&.call(name))
    end

    def validate!(block : HyNomasDualStreamBlock,
                  img : ML::MetalBuffer, txt : ML::MetalBuffer,
                  vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                  video_rope_cos : ML::MetalBuffer,
                  video_rope_sin : ML::MetalBuffer) : Nil
      @workspace.validate_for!(block, @chain, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin)
    end

    def linear(e : ML::Metal::ComputeEncoder, name : String,
               input : ML::MetalBuffer,
               weights : HyNomasDualStreamBlock::LinearWeights,
               output : ML::MetalBuffer, rows : Int32,
               input_dim : Int32, output_dim : Int32) : Nil
      @workspace.project(@chain, name, input, weights, output, rows, input_dim, output_dim)
    end

    def attention(e : ML::Metal::ComputeEncoder,
                  query : ML::MetalBuffer, key : ML::MetalBuffer,
                  value : ML::MetalBuffer, output : ML::MetalBuffer,
                  config : HyNomasConfig) : ML::Metal::ComputeEncoder
      @workspace.encode_attention(@chain, query, key, value, output, config)
      @chain.encoder
    end

    def layer_norm(e : ML::Metal::ComputeEncoder,
                   input : ML::MetalBuffer, output : ML::MetalBuffer,
                   rows : Int32, dim : Int32, epsilon : Float32) : Nil
      @workspace.layer_norm(e, input, output, rows, dim, epsilon)
    end

    def activation(e : ML::Metal::ComputeEncoder, name : String,
                   input : ML::MetalBuffer, output : ML::MetalBuffer) : ML::Metal::ComputeEncoder
      @workspace.activation(@chain, name, input, output)
      @chain.encoder
    end

    def rms_norm(e : ML::Metal::ComputeEncoder, name : String,
                 input : ML::MetalBuffer, weight : HyNomasDualStreamBlock::NormWeight,
                 output : ML::MetalBuffer) : ML::Metal::ComputeEncoder
      @workspace.rms_norm(@chain, name, input, weight, output)
      @chain.encoder
    end

    def rope(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
             output : ML::MetalBuffer, cosine : ML::MetalBuffer,
             sine : ML::MetalBuffer, tokens : Int32) : Nil
      @workspace.rope(e, input, output, cosine, sine, tokens)
    end

    def modulate(e : ML::Metal::ComputeEncoder,
                 input : ML::MetalBuffer, shift : ML::MetalBuffer,
                 scale : ML::MetalBuffer, output : ML::MetalBuffer,
                 rows : Int32, dim : Int32) : Nil
      @workspace.modulate(e, input, shift, scale, output, rows, dim)
    end

    def gated_residual(e : ML::Metal::ComputeEncoder,
                       input : ML::MetalBuffer, delta : ML::MetalBuffer,
                       gate : ML::MetalBuffer, output : ML::MetalBuffer,
                       rows : Int32, dim : Int32) : Nil
      @workspace.gated_residual(e, input, delta, gate, output, rows, dim)
    end
  end

  # Add the opt-in overload only when callers explicitly require this file.
  class HyNomasDualStreamBlock
    # phase_observer is a trusted diagnostic callback, not a sandbox. It must
    # only observe CPU metadata and must not mutate resources, encode, submit
    # or wait. Its overhead belongs to instrumented timings. If it raises after
    # work was appended, the caller must drain the chain before releasing it.
    def encode_mps(chain : ML::Metal::MPSCommandChain,
                   img : ML::MetalBuffer, txt : ML::MetalBuffer,
                   vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                   video_rope_cos : ML::MetalBuffer,
                   video_rope_sin : ML::MetalBuffer,
                   workspace : HyNomasMPSWorkspace,
                   *, phase_observer : Proc(String, Nil)? = nil) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      backend = HyNomasMPSBackend.new(chain, workspace, phase_observer)
      encode_with_backend(chain.encoder, img, txt, vec, vec_txt,
        video_rope_cos, video_rope_sin, backend)
    end
  end
end
