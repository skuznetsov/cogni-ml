require "./hy_nomas_probe_instrumentation"
require "../../src/ml/video/hy_nomas_bf16_stack"
require "../../src/ml/video/hy_nomas_source_bf16_mps"
require "../../src/ml/video/hy_nomas_continuation_history"
require "../../src/ml/video/hy_nomas_dit_boundaries"

# Diagnostic access to the existing registered block/workspace pair. The probe
# computes only the first registered block, with its unchanged fixed bindings;
# the serial owner retains and releases every shared intermediate allocation.
class ML::Video::HyNomasMPSSerialWorkspace
  def dit_route_probe_workspace : ML::Video::HyNomasMPSWorkspace
    workspace
  end
end

module HyNomasDiTRouteFactorialGPU
  extend self
  CASE_ORDER  = ["M11", "M00", "M10", "M01", "M11_repeat"]
  CHECKPOINTS = ["velocity", "block0.image", "block0.text_valid",
                 "block53.image", "block53.text_valid"]
  VIDEO           = 4320
  DIM             = 2048
  TEXT            =  320
  T0_VALID        =  263
  T1_VALID        =  224
  HISTORY         = VIDEO + T0_VALID
  QUERY           = VIDEO + T1_VALID
  HEADS           =  16
  HEAD_DIM        = 128
  BLOCKS          =  54
  IMAGE_ELEMENTS  = VIDEO * DIM
  TEXT_ELEMENTS   = TEXT * DIM
  ROPE_ELEMENTS   = VIDEO * HEAD_DIM
  LATENT_ELEMENTS = VIDEO * 32
  SOURCE_SHA      = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
  CONFIG          = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: VIDEO,
    text_tokens: TEXT, dim: DIM, heads: HEADS, head_dim: HEAD_DIM,
    ffn_dim: 8192,
    attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
  CACHE_CONFIG = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
    text_tokens: 1280, dim: DIM, heads: HEADS, head_dim: HEAD_DIM, ffn_dim: 8192)

  def checkpoint_elements(name : String) : Int32
    case name
    when "velocity"                                then LATENT_ELEMENTS
    when "block0.image", "block53.image"           then IMAGE_ELEMENTS
    when "block0.text_valid", "block53.text_valid" then T1_VALID * DIM
    else                                                raise ArgumentError.new("unexpected direct-route checkpoint #{name}")
    end
  end

  # No Arena, MetalBuffer, pipeline or device is constructed by this estimate.
  # Opaque MPS resources and host/reference storage remain under the external
  # footprint guard, not under this exact named-allocation accounting.
  def memory_plan(action_logit_biases : Hash(String, Array(Float32))? = nil) : Hash(String, Int64)
    bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG)
    stage = bindings.sum(0_i64) { |binding| binding.shape.reduce(1_i64) { |n, d| n * d } * 2 }
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG).sum(0_i64) { |_, n| n.to_i64 * 4 }
    # The existing registration API requires two blocks. The unused second
    # block shares every intermediate and owns only its two final output banks.
    scratch += CONFIG.joint_rows.to_i64 * DIM * 4
    rms = 2_i64 * CONFIG.joint_rows * (DIM + HEADS) * 4
    mps = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(CONFIG).to_i64 * 4 + rms +
          ML::Video::HyNomasSourceBF16MPSWorkspace.extra_bytes(CONFIG)
    t0_attention = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(0, HISTORY,
      HEADS, HEAD_DIM, source_bf16_attention: true).workspace_bytes.to_i64
    # Label values do not change geometry. Global labels are sufficient for
    # this allocation estimate; actual execution authenticates all mask bytes.
    case_order = action_logit_biases ? action_logit_biases.keys : CASE_ORDER
    t1_attention = case_order[0...-1].sum(0_i64) do |name|
      bias = action_logit_biases.try(&.[name])
      bias = nil if bias.try(&.all?(&.zero?))
      ML::Video::HyNomasHistoryAttentionWorkspace.geometry(HISTORY,
        QUERY, HEADS, HEAD_DIM, query_labels: Array(Int32).new(QUERY, 1),
        key_labels: Array(Int32).new(HISTORY + QUERY, 0),
        source_bf16_attention: true, action_logit_bias: bias).workspace_bytes.to_i64
    end
    current = (IMAGE_ELEMENTS.to_i64 + TEXT_ELEMENTS + DIM + 2_i64 * ROPE_ELEMENTS) * 4
    states = (case_order.size.to_i64 + 1) * (IMAGE_ELEMENTS.to_i64 + TEXT_ELEMENTS) * 4
    constants = (2_i64 * DIM + 4_i64 * ROPE_ELEMENTS) * 4
    snapshots = case_order.size.to_i64 * CHECKPOINTS.sum(0_i64) { |name| checkpoint_elements(name).to_i64 * 4 }
    final_weights = ML::Video::HyNomasFinalMPSPlan.checkpoint_bindings.sum(0_i64) do |binding|
      binding.shape.reduce(1_i64) { |n, d| n * d } * 2
    end
    final_scratch = (DIM.to_i64 + 2_i64 * VIDEO * DIM + 2_i64 * DIM +
                     2_i64 * LATENT_ELEMENTS + 2_i64 * DIM * DIM) * 4
    parts = {"resident_block_weights" => BLOCKS.to_i64 * stage, "block_weight_stage" => stage,
             "shared_block_scratch" => scratch, "source_bf16_workspace" => mps,
             "current_inputs" => current, (action_logit_biases ? "state_banks" : "six_state_banks") => states,
             "input_constants" => constants, "snapshots" => snapshots,
             "single_layer_kv" => 2_i64 * HISTORY * DIM * 4,
             "attention_workspaces" => t0_attention + t1_attention,
             "final_weights" => final_weights, "final_workspace" => final_scratch} of String => Int64
    parts["aggregate_named_bytes"] = parts.values.sum(0_i64)
    # Setup briefly has the ordinary compact attention plan before attaching
    # the rectangular history provider. Count that transient peak separately.
    old_attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1,
      HISTORY, HEADS, HEAD_DIM, 256, HEADS).resident_bytes.to_i64
    parts["setup_transient_attention_bytes"] = old_attention
    parts["peak_named_bytes"] = parts["aggregate_named_bytes"] + old_attention
    parts
  end

  # All plans bind the same current Q/K/V/output and the same layer-local T0
  # cache pair. Only their setup-authenticated query labels differ. Selection
  # is CPU metadata; it never edits a live mask or reads an activation.
  class Attention < ML::Video::HyNomasMPSHistoryProvider
    getter cache_key : ML::MetalBuffer
    getter cache_value : ML::MetalBuffer
    @arena : ML::Video::Arena
    @workspaces = [] of ML::Video::HyNomasHistoryAttentionWorkspace
    @plans = {} of String => ML::Video::HyNomasHistoryAttentionMPSPlan
    @selected = "T0"
    @repeat_name = "M11_repeat"
    @baseline_name = "M11"
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@query : ML::MetalBuffer, @key : ML::MetalBuffer,
                   @value : ML::MetalBuffer, @output : ML::MetalBuffer,
                   fixture : HyNomasDiTRouteFactorialProbe::Reference,
                   action_logit_biases : Hash(String, Array(Float32))? = nil)
      @arena = ML::Video::Arena.new
      @cache_key = @arena.alloc("layer.T0.key", HISTORY * DIM)
      @cache_value = @arena.alloc("layer.T0.value", HISTORY * DIM)
      begin
        t0 = ML::Video::HyNomasHistoryAttentionWorkspace.new(0, HISTORY, HEADS,
          HEAD_DIM, source_bf16_attention: true)
        @workspaces << t0
        t0.freeze_history_inventory!
        @plans["T0"] = ML::Video::HyNomasHistoryAttentionMPSPlan.new(t0, @query, @key, @value, @output)
        case_order = action_logit_biases ? action_logit_biases.keys : CASE_ORDER
        @baseline_name = case_order.first
        @repeat_name = case_order.last
        case_order[0...-1].each do |name|
          bias = action_logit_biases.try(&.[name])
          bias = nil if bias.try(&.all?(&.zero?))
          work = ML::Video::HyNomasHistoryAttentionWorkspace.new(HISTORY, QUERY,
            HEADS, HEAD_DIM, query_labels: fixture.native_query_labels_for(action_logit_biases ? "M11" : name),
            key_labels: fixture.key_labels, source_bf16_attention: true,
            action_logit_bias: bias)
          @workspaces << work
          work.register_history_pair!(@cache_key, @cache_value)
          work.freeze_history_inventory!
          @plans[name] = ML::Video::HyNomasHistoryAttentionMPSPlan.new(work, @query, @key, @value, @output)
        end
      rescue ex
        release
        raise ex
      end
    end

    def select!(name : String) : Nil
      raise ArgumentError.new("released route attention") if @released
      key = name == @repeat_name ? @baseline_name : name
      raise ArgumentError.new("unknown route attention case") unless @plans.has_key?(key)
      @selected = key
    end

    def resident_bytes : Int64
      @arena.resident_bytes + @workspaces.sum(&.resident_bytes)
    end

    def validate_bindings!(tokens : Int32, heads : Int32, head_dim : Int32,
                           query : ML::MetalBuffer, key : ML::MetalBuffer,
                           value : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      unless tokens == HISTORY && heads == HEADS && head_dim == HEAD_DIM &&
             query.same?(@query) && key.same?(@key) && value.same?(@value) && output.same?(@output)
        raise ArgumentError.new("compact direct-route attention binding changed")
      end
      @plans.each_value { |plan| plan.validate_bindings!(query, key, value, output) }
    end

    def validate_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("released route attention") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("route attention cannot overlap command chains")
        end
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain, block_index : Int32) : Nil
      raise ArgumentError.new("only the first registered block is used by the route probe") unless block_index == 0
      validate_chain!(chain)
      @last_chain = chain
      @plans[@selected].encode(chain, @selected == "T0" ? -1 : 0)
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("drain route attention before release") unless chain.completed?
      end
      @plans.each_value(&.release)
      @plans.clear
      @workspaces.reverse_each(&.release)
      @workspaces.clear
      @arena.release
      @released = true
    end
  end

  class Runner
    @inputs : ML::Video::Arena
    @states : ML::Video::Arena
    @snapshots : ML::Video::Arena
    @stage : ML::Video::Arena
    @globals : ML::Video::Arena
    @loader : ML::Video::HyNomasBF16Stack
    @serial : ML::Video::HyNomasMPSSerialWorkspace
    @attention : Attention
    @final : ML::Video::HyNomasFinalMPSPlan
    @prepared : ML::Video::PreparedWeights
    @global_file : ML::Video::CheckpointFile
    @block : ML::Video::HyNomasDualStreamBlock
    @workspace : ML::Video::HyNomasMPSWorkspace
    @chain : ML::Metal::MPSCommandChain? = nil
    @wait_attempted = false
    @released = false
    getter plan : Hash(String, Int64)
    @case_order : Array(String)

    def initialize(fixture : HyNomasDiTRouteFactorialProbe::Reference, budget : Int64,
                   action_logit_biases : Hash(String, Array(Float32))? = nil)
      @case_order = action_logit_biases ? action_logit_biases.keys : CASE_ORDER
      if biases = action_logit_biases
        unless @case_order.size >= 2 && biases[@case_order.first].all?(&.zero?) &&
               biases[@case_order.last] == biases[@case_order.first]
          raise ArgumentError.new("soft-route runner requires zero baseline and exact final repeat")
        end
        @case_order.each do |name|
          unless biases[name].size == QUERY
            raise ArgumentError.new("soft-route bias row count differs from compact queries")
          end
        end
      end
      @plan = HyNomasDiTRouteFactorialGPU.memory_plan(action_logit_biases)
      raise ArgumentError.new("direct-route peak named buffers exceed budget") if @plan["peak_named_bytes"] > budget
      admission = ML::Video::HyNomasQ4StackIndex::Admission.new(fixture.cache_index_path,
        fixture.cache_index_sha256, SOURCE_SHA, CACHE_CONFIG)
      @global_file = ML::Video::BF16WeightCache.authenticate(admission.globals_bf16_path,
        admission.globals_bf16_sha256)
      final_bindings = ML::Video::HyNomasFinalMPSPlan.checkpoint_bindings.map do |binding|
        ML::Video::WeightBinding.new(binding.target, binding.source, binding.shape,
          ML::GGUF::SafeTensorDType::BF16)
      end
      weight_plan = ML::Video::WeightPlan.new([@global_file], final_bindings, budget,
        resident_dtype: ML::Video::WeightDType::BF16)
      unless weight_plan.destination_bytes == @plan["final_weights"] && weight_plan.staging_bytes == 0
        raise ArgumentError.new("final BF16 subset differs from named plan")
      end
      @loader = ML::Video::HyNomasBF16Stack.new(admission, fixture.cache_index_sha256,
        SOURCE_SHA, CACHE_CONFIG, budget, stage_bytes: @plan["block_weight_stage"],
        reserve_bytes: @plan["peak_named_bytes"] - @plan["resident_block_weights"] - @plan["block_weight_stage"])
      @inputs = ML::Video::Arena.new
      @states = ML::Video::Arena.new
      @snapshots = ML::Video::Arena.new
      @stage = ML::Video::Arena.new
      @globals = ML::Video::Arena.new
      @serial = uninitialized ML::Video::HyNomasMPSSerialWorkspace
      @attention = uninitialized Attention
      @final = uninitialized ML::Video::HyNomasFinalMPSPlan
      @prepared = uninitialized ML::Video::PreparedWeights
      @block = uninitialized ML::Video::HyNomasDualStreamBlock
      @workspace = uninitialized ML::Video::HyNomasMPSWorkspace
      cleanup = [] of Proc(Nil)
      begin
        @prepared = ML::Video::PreparedWeights.new(weight_plan, @globals)
        cleanup << -> { @prepared.release }
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG).each do |binding|
          @stage.alloc_weight(binding.target, binding.shape.reduce(1_i64) { |n, d| n * d }.to_i32,
            ML::Video::WeightDType::BF16)
        end
        @loader.load!(@stage)
        img = @inputs.alloc("working.image", IMAGE_ELEMENTS)
        txt = @inputs.alloc("working.text", TEXT_ELEMENTS)
        vec = @inputs.alloc("working.vec", DIM)
        cos = @inputs.alloc("working.cos", ROPE_ELEMENTS)
        sin = @inputs.alloc("working.sin", ROPE_ELEMENTS)
        @inputs.upload("T0.vec", fixture.inputs["t0.vector"])
        @inputs.upload("T1.vec", fixture.inputs["t1.step1.vector"])
        {"cos" => fixture.rope_cos, "sin" => fixture.rope_sin}.each do |kind, values|
          raise ArgumentError.new("expected exactly two frozen RoPE banks") unless values.size == 2 * ROPE_ELEMENTS
          @inputs.upload("T0.#{kind}", values[0, ROPE_ELEMENTS])
          @inputs.upload("T1.#{kind}", values[ROPE_ELEMENTS, ROPE_ELEMENTS])
        end
        (["T0"] + @case_order).each do |name|
          prefix = name == "T0" ? "t0" : "t1.step1"
          @states.upload("#{name}.image", fixture.inputs["#{prefix}.patch"])
          padded = Array(Float32).new(TEXT_ELEMENTS, 0.0_f32)
          values = fixture.inputs["#{prefix}.text_valid"]
          values.each_with_index { |value, index| padded[index] = value }
          @states.upload("#{name}.text", padded)
        end
        @case_order.each do |name|
          CHECKPOINTS.each do |checkpoint|
            @snapshots.alloc("#{name}.#{checkpoint}", HyNomasDiTRouteFactorialGPU.checkpoint_elements(checkpoint))
          end
        end
        @serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG, [@stage, @stage],
          img, txt, vec, vec, cos, sin, attention_query_chunk: 256,
          attention_heads_per_group: HEADS, valid_text_tokens: T0_VALID,
          allow_shared_modulation: true,
          workspace_factory: ML::Video::HyNomasSourceBF16MPSWorkspace.factory)
        cleanup << -> { @serial.release }
        @block = @serial.blocks.first
        @workspace = @serial.dit_route_probe_workspace
        joint = @block.checkpoints
        @attention = Attention.new(joint["joint.query"], joint["joint.key"],
          joint["joint.value"], joint["joint.attention"], fixture, action_logit_biases)
        cleanup << -> { @attention.release }
        @serial.attach_history_provider!(@attention)
        @final = ML::Video::HyNomasFinalMPSPlan.new(img, vec, VIDEO, @globals,
          source_bf16: true, source_autocast: true)
        cleanup << -> { @final.release }
        actual = @loader.cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes +
                 @inputs.resident_bytes + @states.resident_bytes + @snapshots.resident_bytes +
                 @serial.resident_bytes + @final.resident_bytes
        unless actual == @plan["aggregate_named_bytes"]
          raise ArgumentError.new("direct-route named plan mismatch: #{actual} != #{@plan["aggregate_named_bytes"]}")
        end
      rescue ex
        cleanup.reverse_each(&.call)
        @loader.release
        [@inputs, @states, @snapshots, @stage, @globals].each(&.release)
        @global_file.close
        @released = true
        raise ex
      end
    end

    def run : NamedTuple(outputs: Hash(String, Array(Float32)), encode_allocations: Int64,
      encode_readbacks: Int64, terminal_waits: Int32, seconds: Float64)
      raise ArgumentError.new("route runner is single-use") if @chain || @released
      chain = ML::Metal::MPSCommandChain.new
      @chain = chain
      before_alloc = ML::MetalBuffer.hy_text_probe_allocations
      before_reads = ML::MetalBuffer.hy_text_probe_readbacks
      started = Time.instant
      @prepared.encode(chain.encoder)
      BLOCKS.times do |layer|
        @loader.encode!(chain, layer, @stage)
        encode_state(chain, "T0", "T0")
        # These Q/K/V checkpoints still belong to T0. Cache them before the
        # first arm overwrites the common scratch; only this layer is retained.
        joint = @block.checkpoints
        copy(chain, joint["joint.key"], @attention.cache_key, HISTORY * DIM)
        copy(chain, joint["joint.value"], @attention.cache_value, HISTORY * DIM)
        @case_order.each do |name|
          encode_state(chain, name, "T1")
          if layer == 0 || layer == BLOCKS - 1
            copy(chain, @block.image_output, @snapshots["#{name}.block#{layer}.image"], IMAGE_ELEMENTS)
            copy(chain, @block.text_output, @snapshots["#{name}.block#{layer}.text_valid"], T1_VALID * DIM)
          end
        end
      end
      @case_order.each do |name|
        copy(chain, @states["#{name}.image"], @inputs["working.image"], IMAGE_ELEMENTS)
        copy(chain, @inputs["T1.vec"], @inputs["working.vec"], DIM)
        @final.encode(chain)
        copy(chain, @final.output, @snapshots["#{name}.velocity"], LATENT_ELEMENTS)
      end
      allocations = ML::MetalBuffer.hy_text_probe_allocations - before_alloc
      reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
      raise "direct-route encode allocated named GPU buffers" unless allocations == 0
      raise "direct-route encode read back an activation" unless reads == 0
      raise "direct-route encode introduced an intermediate wait" unless chain.wait_count == 0
      @wait_attempted = true
      chain.commit_and_wait
      raise "direct-route chain did not have one terminal wait" unless chain.completed? && chain.wait_count == 1
      seconds = (Time.instant - started).total_seconds
      outputs = {} of String => Array(Float32)
      @case_order.each do |name|
        CHECKPOINTS.each do |checkpoint|
          key = "#{name}.#{checkpoint}"
          outputs[key] = @snapshots[key].read(HyNomasDiTRouteFactorialGPU.checkpoint_elements(checkpoint))
        end
      end
      {outputs: outputs, encode_allocations: allocations, encode_readbacks: reads,
       terminal_waits: chain.wait_count, seconds: seconds}
    ensure
      if active = @chain
        unless active.completed? || @wait_attempted
          @wait_attempted = true
          active.commit_and_wait
        end
      end
    end

    private def copy(chain : ML::Metal::MPSCommandChain, source : ML::MetalBuffer,
                     destination : ML::MetalBuffer, count : Int32) : Nil
      ML::Video::MetalOps.copy(chain.encoder, source, destination, count)
    end

    private def encode_state(chain : ML::Metal::MPSCommandChain, name : String, phase : String) : Nil
      img = @inputs["working.image"]
      txt = @inputs["working.text"]
      vec = @inputs["working.vec"]
      cos = @inputs["working.cos"]
      sin = @inputs["working.sin"]
      copy(chain, @states["#{name}.image"], img, IMAGE_ELEMENTS)
      copy(chain, @states["#{name}.text"], txt, TEXT_ELEMENTS)
      copy(chain, @inputs["#{phase}.vec"], vec, DIM)
      copy(chain, @inputs["#{phase}.cos"], cos, ROPE_ELEMENTS)
      copy(chain, @inputs["#{phase}.sin"], sin, ROPE_ELEMENTS)
      @attention.select!(name)
      @block.encode_mps(chain, img, txt, vec, vec, cos, sin, @workspace)
      copy(chain, @block.image_output, @states["#{name}.image"], IMAGE_ELEMENTS)
      copy(chain, @block.text_output, @states["#{name}.text"], TEXT_ELEMENTS)
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("drain direct-route runner before release") unless chain.completed?
      end
      @final.release
      @serial.release
      @loader.release
      @prepared.release
      [@inputs, @states, @snapshots, @stage, @globals].each(&.release)
      @global_file.close
      @chain.try(&.release)
      @released = true
    end
  end
end
