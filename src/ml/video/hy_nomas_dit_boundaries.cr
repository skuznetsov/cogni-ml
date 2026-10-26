require "./hy_nomas_text_refiner"

module ML::Video
  # Setup-owned input/final adapters for the first-segment, camera-off DiT.
  # Every neural operation is appended to the existing chain. Host access is
  # confined to checkpoint setup and diagnostics after its terminal wait.
  abstract class HyNomasDiTBoundaryMPSPlan
    SOURCE = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_dit_boundaries.metal") }}
    private record Linear, input : ML::MetalBuffer, output : ML::MetalBuffer,
      weight : ML::MetalBuffer, bias : ML::MetalBuffer,
      weight_dtype : WeightDType, bias_dtype : WeightDType,
      rows : Int32, input_dim : Int32, output_dim : Int32,
      plan : WanT5MPSLinearPlan

    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of HyNomasDiTBoundaryMPSPlan
    @arena : Arena
    @linears = {} of String => Linear
    @activations = [] of HyNomasMPSActivation::UnaryPlan
    @weights : Arena
    @external = [] of Tuple(String, ML::MetalBuffer, WeightDType, Int64)
    @inputs = [] of Tuple(ML::MetalBuffer, Int64)
    @owned = [] of Tuple(String, ML::MetalBuffer, Int64)
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    protected def initialize(@weights : Arena)
      @arena = Arena.new
      MetalLinearMPS.prepare!
    end

    def resident_bytes : Int64
      ensure_live!
      @arena.resident_bytes
    end

    # Read-only preflight may be called for the entire request before any
    # producer appends commands. It never binds a new chain or mutates buffers.
    def validate_for!(chain : ML::Metal::MPSCommandChain) : Nil
      ensure_live!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("DiT boundary cannot span in-flight command chains")
        end
      end
      @inputs.each { |buffer, bytes| validate_f32!(buffer, bytes) }
      @external.each do |name, buffer, dtype, bytes|
        unless @weights.allocated?(name) && @weights[name].same?(buffer) &&
               @weights.dtype(name) == dtype && buffer.valid? &&
               buffer.size == bytes && buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("DiT boundary weight binding changed: #{name}")
        end
      end
      @owned.each do |name, buffer, bytes|
        unless @arena.allocated?(name) && @arena[name].same?(buffer)
          raise ArgumentError.new("DiT boundary scratch binding changed: #{name}")
        end
        validate_f32!(buffer, bytes)
      end
      @linears.each_value do |entry|
        MetalLinearMPS.validate_preexpanded!(entry.input, entry.bias, entry.output,
          entry.rows, entry.input_dim, entry.output_dim, entry.bias_dtype,
          entry.plan, @arena["weight_stage"])
      end
    end

    def release : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the DiT boundary chain before release") unless previous.completed?
      end
      @activations.each(&.release)
      @activations.clear
      @linears.each_value(&.plan.release)
      @linears.clear
      @arena.release
      @released = true
      @@lifetime_mutex.synchronize { @@live_plans.reject! { |plan| plan.same?(self) } }
    end

    protected def finish_setup! : Nil
      @owned = @arena.buffers.map { |name, buffer| {name, buffer, buffer.size} }
      @@lifetime_mutex.synchronize { @@live_plans << self }
    end

    protected def bind!(chain : ML::Metal::MPSCommandChain) : Nil
      validate_for!(chain)
      @last_chain = chain
    end

    protected def ensure_live! : Nil
      raise ArgumentError.new("DiT boundary plan has been released") if @released
    end

    protected def input!(buffer : ML::MetalBuffer, count : Int64) : Nil
      validate_f32!(buffer, count * 4)
      @inputs << {buffer, count * 4}
    end

    protected def validate_f32!(buffer : ML::MetalBuffer, bytes : Int64) : Nil
      unless bytes > 0 && buffer.valid? && buffer.size == bytes &&
             buffer.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("DiT boundary requires an exact shared F32 buffer")
      end
    end

    protected def add_linear!(name : String, input : ML::MetalBuffer,
                              output : ML::MetalBuffer, rows : Int32,
                              input_dim : Int32, output_dim : Int32) : Nil
      weight_name, bias_name = "#{name}.weight", "#{name}.bias"
      weight, bias = @weights[weight_name], @weights[bias_name]
      weight_dtype, bias_dtype = @weights.dtype(weight_name), @weights.dtype(bias_name)
      MetalOps.check_storage(weight, input_dim.to_i64 * output_dim, weight_dtype)
      MetalOps.check_storage(bias, output_dim.to_i64, bias_dtype)
      unless weight.size == input_dim.to_i64 * output_dim * weight_dtype.bytes_per_element &&
             bias.size == output_dim.to_i64 * bias_dtype.bytes_per_element
        raise ArgumentError.new("DiT boundary projection has non-exact weight storage")
      end
      @external << {weight_name, weight, weight_dtype, weight.size}
      @external << {bias_name, bias, bias_dtype, bias.size}
      MetalOps.distinct(output, input, weight, bias)
      MetalOps.distinct(@arena["weight_stage"], input, output, weight, bias)
      cfg = WanT5Config.new(vocab: 1, dim: input_dim, ffn_dim: output_dim,
        heads: input_dim, layers: 1, batches: 1, tokens: rows)
      plan = WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, @arena["weight_stage"], output)
      @linears[name] = Linear.new(input, output, weight, bias, weight_dtype,
        bias_dtype, rows, input_dim, output_dim, plan)
    end

    protected def project!(encoder : ML::Metal::ComputeEncoder, name : String) : Nil
      entry = @linears[name]
      MetalLinearMPS.project(encoder, entry.input, entry.weight, entry.bias,
        entry.output, entry.rows, entry.input_dim, entry.output_dim,
        entry.weight_dtype, entry.bias_dtype, entry.plan, @arena["weight_stage"])
    end

    protected def pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end
  end

  # Patch [1,1,1]: an exact layout transpose followed by the Conv3d pointwise
  # weight interpreted as row-major [out,in]. Flattening preserves T,H,W order.
  class HyNomasPatchMPSPlan < HyNomasDiTBoundaryMPSPlan
    CHANNELS =   32
    DIM      = 2048
    @input : ML::MetalBuffer
    @tokens : Int32
    @transpose : ML::Metal::ComputePipeline

    def self.checkpoint_bindings : Array(WeightBinding)
      [WeightBinding.new("img_in.proj.weight", "img_in.proj.weight", [DIM.to_i64, CHANNELS.to_i64, 1_i64, 1_i64, 1_i64]),
       WeightBinding.new("img_in.proj.bias", "img_in.proj.bias", [DIM.to_i64])]
    end

    def initialize(@input : ML::MetalBuffer, @tokens : Int32, weights : Arena)
      raise ArgumentError.new("DiT patch token count must be in 1..5280") unless 0 < @tokens <= 5280
      @transpose = pipeline("hy_nomas_dit_channels_to_tokens_f32")
      super(weights)
      begin
        input!(@input, @tokens.to_i64 * CHANNELS)
        @arena.alloc("tokens", @tokens * CHANNELS)
        @arena.alloc("output", @tokens * DIM)
        @arena.alloc("weight_stage", DIM * CHANNELS)
        add_linear!("img_in.proj", @arena["tokens"], @arena["output"], @tokens, CHANNELS, DIM)
        finish_setup!
      rescue ex
        release
        raise ex
      end
    end

    def output : ML::MetalBuffer
      ensure_live!
      @arena["output"]
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      bind!(chain)
      e = chain.encoder
      e.set_pipeline(@transpose).set_buffer(@input, 0)
        .set_buffer(@arena["tokens"], 1, ML::Metal::BufferAccess::Write)
        .set_value(@tokens.to_u32, 2).set_value(CHANNELS.to_u32, 3)
        .dispatch_1d(@tokens * CHANNELS)
      project!(e, "img_in.proj")
    end
  end

  class HyNomasFinalMPSPlan < HyNomasDiTBoundaryMPSPlan
    DIM      = 2048
    CHANNELS =   32
    @input : ML::MetalBuffer
    @vec : ML::MetalBuffer
    @tokens : Int32
    @norm : HyNomasMPSLayerNorm::Plan
    @silu : HyNomasMPSActivation::SiLUPlan? = nil
    @modulate : ML::Metal::ComputePipeline
    @unpatch : ML::Metal::ComputePipeline

    def self.checkpoint_bindings : Array(WeightBinding)
      [WeightBinding.new("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.weight", [2_i64 * DIM, DIM.to_i64]),
       WeightBinding.new("final_layer.adaLN_modulation.1.bias", "final_layer.adaLN_modulation.1.bias", [2_i64 * DIM]),
       WeightBinding.new("final_layer.linear.weight", "final_layer.linear.weight", [CHANNELS.to_i64, DIM.to_i64]),
       WeightBinding.new("final_layer.linear.bias", "final_layer.linear.bias", [CHANNELS.to_i64])]
    end

    def initialize(@input : ML::MetalBuffer, @vec : ML::MetalBuffer,
                   @tokens : Int32, weights : Arena)
      raise ArgumentError.new("DiT final token count must be in 1..5280") unless 0 < @tokens <= 5280
      @norm = HyNomasMPSLayerNorm::Plan.new(DIM)
      @modulate = pipeline("hy_nomas_dit_final_modulate_f32")
      @unpatch = pipeline("hy_nomas_dit_tokens_to_channels_f32")
      super(weights)
      begin
        input!(@input, @tokens.to_i64 * DIM)
        input!(@vec, DIM.to_i64)
        @arena.alloc("silu", DIM)
        silu = HyNomasMPSActivation::SiLUPlan.new([1, DIM], @vec, @arena["silu"])
        @silu = silu
        @activations << silu
        @arena.alloc("norm", @tokens * DIM)
        @arena.alloc("modulation", 2 * DIM)
        @arena.alloc("modulated", @tokens * DIM)
        @arena.alloc("linear", @tokens * CHANNELS)
        @arena.alloc("output", @tokens * CHANNELS)
        @arena.alloc("weight_stage", 2 * DIM * DIM)
        add_linear!("final_layer.adaLN_modulation.1", @arena["silu"], @arena["modulation"], 1, DIM, 2 * DIM)
        add_linear!("final_layer.linear", @arena["modulated"], @arena["linear"], @tokens, DIM, CHANNELS)
        finish_setup!
      rescue ex
        release
        raise ex
      end
    end

    def checkpoints : Hash(String, ML::MetalBuffer)
      ensure_live!
      {"final.norm" => @arena["norm"], "final.modulation" => @arena["modulation"],
       "final.modulated" => @arena["modulated"], "final.linear" => @arena["linear"],
       "output.latent" => @arena["output"]}
    end

    def output : ML::MetalBuffer
      ensure_live!
      @arena["output"]
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      bind!(chain)
      @silu.not_nil!.encode(chain)
      project!(chain.encoder, "final_layer.adaLN_modulation.1")
      @norm.encode(chain.encoder, @input, @arena["norm"], @tokens)
      chain.encoder.set_pipeline(@modulate).set_buffer(@arena["norm"], 0)
        .set_buffer(@arena["modulation"], 1)
        .set_buffer(@arena["modulated"], 2, ML::Metal::BufferAccess::Write)
        .set_value(DIM.to_u32, 3).set_value((@tokens * DIM).to_u32, 4)
        .dispatch_1d(@tokens * DIM)
      project!(chain.encoder, "final_layer.linear")
      chain.encoder.set_pipeline(@unpatch).set_buffer(@arena["linear"], 0)
        .set_buffer(@arena["output"], 1, ML::Metal::BufferAccess::Write)
        .set_value(@tokens.to_u32, 2).set_value(CHANNELS.to_u32, 3)
        .dispatch_1d(@tokens * CHANNELS)
    end
  end
end
