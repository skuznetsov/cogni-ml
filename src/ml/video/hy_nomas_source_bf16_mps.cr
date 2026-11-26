# Opt-in source-BF16 arithmetic for the serial Hy-Nomas DiT.
require "./hy_nomas_serial_mps"
require "./hy_nomas_mps_bf16_linear"
require "./hy_nomas_reference_bf16"

module ML::Video
  class HyNomasSourceBF16MPSWorkspace < HyNomasMPSWorkspace
    private alias ProjectionKey = Tuple(String, ML::MetalBuffer, ML::MetalBuffer)

    @source_arena = Arena.new
    @source_projections = {} of ProjectionKey => HyNomasMPSBF16Linear::Plan
    @source_silu = {} of String => Tuple(HyNomasMPSActivation::SiLUPlan, ML::MetalBuffer)
    @source_gelu = {} of String => Tuple(HyNomasMPSActivation::BF16GELUPlan, ML::MetalBuffer)
    @norm_inputs = {} of ML::MetalBuffer => ML::MetalBuffer
    @raw_projections = {} of String => ML::MetalBuffer
    @source_pointwise : HyNomasReferenceBF16::Plan
    @expand_pipeline : ML::Metal::ComputePipeline
    @residual_pipeline : ML::Metal::ComputePipeline

    def self.factory : HyNomasMPSSerialWorkspace::WorkspaceFactory
      ->(block : HyNomasDualStreamBlock, buffers : Array(ML::MetalBuffer), chunk : Int32, group : Int32, shared : Bool, valid_text : Int32?, _rms_text : Int32?) do
        img, txt, vec, vec_txt, cos, sin = buffers
        new(block, img, txt, vec, vec_txt, cos, sin, chunk, group, shared, valid_text)
          .as(HyNomasMPSWorkspace)
      end
    end

    def self.validate_config!(config : HyNomasConfig) : Nil
      config.validate!
      unless config.batches == 1 && config.effective_img_modulation_rows == 1 &&
             config.effective_txt_modulation_rows == 1 && config.attention_precision.bf16_boundary_control?
        raise ArgumentError.new("Hy-Nomas source BF16 requires batch one, global modulation and BF16 packing")
      end
    end

    def self.extra_bytes(config : HyNomasConfig) : Int64
      validate_config!(config)
      rows = config.image_rows.to_i64 + config.text_rows
      dim = config.dim.to_i64
      # Rounded norm and SiLU inputs in F32. Each stream has physical BF16
      # modulation, projection, FC1 and GELU storage.
      (rows + 2) * dim * 4 + (12 * dim + rows * (dim + 2_i64 * config.ffn_dim)) * 2
    end

    private def initialize(block : HyNomasDualStreamBlock,
                           img : ML::MetalBuffer, txt : ML::MetalBuffer,
                           vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                           cos : ML::MetalBuffer, sin : ML::MetalBuffer,
                           chunk : Int32, group : Int32, shared : Bool,
                           valid_text : Int32?)
      c = block.config
      HyNomasSourceBF16MPSWorkspace.validate_config!(c)
      @source_pointwise = HyNomasReferenceBF16::Plan.new(c.dim)
      @expand_pipeline = ML::Metal::PipelineCache.get("hy_nomas_source_bf16_expand") do
        ML::Metal::ComputePipeline.new("hy_nomas_source_bf16_expand", HyNomasReferenceBF16::SOURCE)
      end
      @residual_pipeline = ML::Metal::PipelineCache.get("hy_nomas_source_bf16_residual") do
        ML::Metal::ComputePipeline.new("hy_nomas_source_bf16_residual", HyNomasReferenceBF16::SOURCE)
      end
      super(block, img, txt, vec, vec_txt, cos, sin, attention_query_chunk: chunk,
        attention_heads_per_group: group, allow_shared_modulation: shared,
        valid_text_tokens: valid_text)
      begin
        checkpoints = block.checkpoints
        {"img" => c.image_rows, "txt" => c.text_rows}.each do |stream, rows|
          name = "#{stream}.attn.norm"
          @norm_inputs[checkpoints[name]] = @source_arena.alloc("#{name}.input",
            HyNomasDualStreamBlock.checked_count(rows, c.dim))
          {"modulation" => HyNomasDualStreamBlock.checked_count(6, c.dim),
           "projection" => HyNomasDualStreamBlock.checked_count(rows, c.dim),
           "ff.fc1"     => HyNomasDualStreamBlock.checked_count(rows, c.ffn_dim),
           "gelu"       => HyNomasDualStreamBlock.checked_count(rows, c.ffn_dim)}.each do |role, count|
            key = "#{stream}.#{role}"
            @raw_projections[key] = @source_arena.alloc_weight(key, count, WeightDType::BF16)
          end
        end
        block.mps_activation_operations(vec, vec_txt).each do |op|
          if op.name.ends_with?(".silu")
            rounded = @source_arena.alloc("#{op.name}.input", c.dim)
            @source_silu[op.name] = {HyNomasMPSActivation::SiLUPlan.new(op.shape, rounded, op.output), rounded}
          else
            stream = op.name.split('.', 2).first
            raw = @raw_projections[op.name]
            @source_gelu[op.name] = {HyNomasMPSActivation::BF16GELUPlan.new(op.shape,
              @raw_projections["#{stream}.ff.fc1"], raw), raw}
          end
        end
        unless @source_arena.resident_bytes == self.class.extra_bytes(c)
          raise ArgumentError.new("Hy-Nomas source BF16 workspace allocation differs from admission")
        end
      rescue ex
        release
        raise ex
      end
    end

    def register_serial_blocks!(blocks : Array(HyNomasDualStreamBlock),
                                inputs : Array(Tuple(ML::MetalBuffer, ML::MetalBuffer)),
                                *, full_dit : Bool = false) : Nil
      super
      blocks.each do |block|
        block.mps_linear_operations.each do |op|
          key = projection_key(op.name, op.weights)
          next if @source_projections.has_key?(key)
          stream, role = op.name.split('.', 2)
          # Every projection widens its BF16 result into a dedicated F32
          # checkpoint before the next encode. Only FC1 must stay physical
          # until its GELU; all width-D projections can reuse the raw bank.
          raw_role = {"modulation", "ff.fc1"}.includes?(role) ? role : "projection"
          @source_projections[key] = HyNomasMPSBF16Linear::Plan.new(op.input, op.weights,
            @raw_projections["#{stream}.#{raw_role}"], op.output,
            HyNomasMPSBF16Linear::Variant::BF16GraphOps, op.rows, op.input_dim, op.output_dim)
        end
      end
    end

    def resident_bytes : Int64
      super + @source_arena.resident_bytes
    end

    protected def source_bf16_rms? : Bool
      true
    end

    def project(chain : ML::Metal::MPSCommandChain, name : String,
                input : ML::MetalBuffer, weights : HyNomasDualStreamBlock::LinearWeights,
                output : ML::MetalBuffer, rows : Int32, input_dim : Int32, output_dim : Int32) : Nil
      validate_projection!(chain, name, input, weights, output, rows, input_dim, output_dim)
      @source_projections[projection_key(name, weights)].encode(chain)
    end

    def activation(chain : ML::Metal::MPSCommandChain, name : String,
                   input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Hy-Nomas MPS workspace has been released") if @released
      op = @activation_operations[name]?
      unless op && op.input.same?(input) && op.output.same?(output)
        raise ArgumentError.new("Hy-Nomas source BF16 activation binding changed: #{name}")
      end
      count = (output.size // sizeof(Float32)).to_i32
      if silu = @source_silu[name]?
        plan, rounded = silu
        @source_pointwise.round(chain.encoder, input, rounded, count)
        plan.encode(chain)
        @source_pointwise.round(chain.encoder, output, output, count)
      else
        plan, raw = @source_gelu[name]
        plan.encode(chain)
        chain.encoder.set_pipeline(@expand_pipeline).set_buffer(raw, 0)
          .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
          .set_value(count.to_u32, 2).dispatch_1d(count)
      end
    end

    def layer_norm(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                   output : ML::MetalBuffer, rows : Int32, dim : Int32, epsilon : Float32) : Nil
      if rounded = @norm_inputs[output]?
        @source_pointwise.round(e, input, rounded, HyNomasDualStreamBlock.checked_count(rows, dim))
        input = rounded
      end
      super(e, input, output, rows, dim, epsilon)
    end

    def modulate(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer, shift : ML::MetalBuffer,
                 scale : ML::MetalBuffer, output : ML::MetalBuffer, rows : Int32, dim : Int32) : Nil
      validate_pointwise_geometry!(rows, dim)
      validate_pointwise_bindings!(input, shift, scale, output)
      @source_pointwise.modulate(e, input, shift, scale, output, rows, dim)
    end

    def rope(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer, output : ML::MetalBuffer,
             cosine : ML::MetalBuffer, sine : ML::MetalBuffer, tokens : Int32) : Nil
      super
      @source_pointwise.round(e, output, output, HyNomasDualStreamBlock.checked_count(tokens, config.dim))
    end

    def gated_residual(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer, delta : ML::MetalBuffer,
                       gate : ML::MetalBuffer, output : ML::MetalBuffer, rows : Int32, dim : Int32) : Nil
      validate_pointwise_geometry!(rows, dim)
      validate_pointwise_bindings!(input, delta, gate, output)
      count = HyNomasDualStreamBlock.checked_count(rows, dim)
      e.set_pipeline(@residual_pipeline).set_buffer(input, 0).set_buffer(delta, 1).set_buffer(gate, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 4).dispatch_1d(count)
    end

    def release : Nil
      super
      @source_silu.each_value { |plan, _| plan.release }
      @source_gelu.each_value { |plan, _| plan.release }
      @source_projections.each_value(&.release)
      @source_silu.clear
      @source_gelu.clear
      @source_projections.clear
      @source_arena.release
    end

    private def projection_key(name : String, weights : HyNomasDualStreamBlock::LinearWeights) : ProjectionKey
      {name, weights.weight, weights.bias}
    end
  end
end
