require "json"
require "digest/sha256"
require "path"
require "../../src/ml/video/hy_nomas_text_refiner"
require "../../src/ml/video/hy_nomas_euler"
require "../../src/ml/video/hy_nomas_dit_boundaries"
require "../../src/ml/video/hy_nomas_continuation_history"
require "../../src/ml/video/hy_nomas_source_bf16_mps"
require "./hy_nomas_generation"
require "./hy_nomas_dit_route_factorial_probe"
require "./hy_nomas_dit_route_factorial_gpu"
require "./hy_nomas_soft_route_probe"

# Compact four-step replay. Every arm keeps the accepted hard M11 support;
# only its video-query/action-text logit bias changes.
module HyNomasSoftRouteRender
  extend self

  SCHEMA                        = "hy-nomas-soft-route-render-native-v1"
  CASES                         = ["zero", "A1_high", "A2_high"]
  OUTER_STEPS                   = 4
  WEIGHT_STAGE_PASSES_PER_CHAIN = OUTER_STEPS
  LOGICAL_FULL_FORWARD_CALLS    = OUTER_STEPS * (CASES.size + 1) # T0 plus each T1 arm per step.
  TENSOR_ORDER                  = CASES.flat_map { |name| (1..OUTER_STEPS).map { |step| "#{name}.step#{step}.latent" } } +
                                  CASES.map { |name| "#{name}.step1.velocity" } +
                                  ["zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector"]
  VIDEO         = 4_320
  DIM           = 2_048
  TEXT          =   320
  T0_VALID      =   263
  T1_VALID      =   224
  HISTORY       = VIDEO + T0_VALID
  QUERY         = VIDEO + T1_VALID
  HEADS         =  16
  HEAD_DIM      = 128
  BLOCKS        =  54
  IMAGE         = VIDEO * DIM
  TEXT_STORAGE  = TEXT * DIM
  VALID_TEXT    = T1_VALID * DIM
  ROPE          = VIDEO * HEAD_DIM
  LATENT        = VIDEO * 32
  TENSOR_SHAPES = TENSOR_ORDER.to_h do |name|
    shape = if name == "zero.step1.patch"
              [1, VIDEO, DIM]
            elsif name == "zero.step1.text_valid"
              [1, T1_VALID, DIM]
            elsif name == "zero.step1.vector"
              [1, DIM]
            else
              [1, 32, 30, 9, 16]
            end
    {name, shape}
  end
  UPSTREAM_SHA             = HyNomasDiTRouteFactorialProbe::CHECKPOINT_SHA256
  PARENT_SHA               = "efcf746fd334218d25b0a922647f10962c6d5102144f7d80cff67bb327a40c89"
  BIAS_SHA                 = "81dd6c8d9ba01bd8824cba8c0c48f238f9180c2cb22a3f18f7c2cb2d5331235e"
  ROI_SOURCE_SHA           = HyNomasDiTRouteFactorialProbe::ROI_MANIFEST_SHA256
  ANCHOR_SHA               = "4a7e2cc71586b0380735f6301c382efcae7ba7fc6ed3eaa0c43b37eb4b57c3c5"
  QWEN_SHA                 = "673d17f11549d89bf5bb4010b22cd9e3e4923ed3071b55a0c7e7348c4e50faf9"
  DEFAULT_PARENT           = "/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/source-qualified.json"
  DEFAULT_ANCHOR           = "/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/anchor/conditioning/t1-inputs-anchor.json"
  DEFAULT_BIAS             = "/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/bias-v2.json"
  DEFAULT_BIAS_PAYLOAD_SHA = BIAS_SHA
  ACCEPTED_FINAL_PATH      = "/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/roi/native/result/t1-step4.f32le"
  ACCEPTED_FINAL_SHA       = "a66c0c3e3de3f070b7f74ff54a4316b04c3860c2385efb830f15f17490561df5"
  NOISE_SELECTION_SHA      = "8abfe96dc1a2b8ee3cd47f5bfac606c5f25630d95c637b535ec7811b122107ef"
  ANCHOR_LATENT_SHA        = "61a5ed5ec852b8abf9c3d799df4fca420eae5e320adb7596d2da0a4c1670cf33"
  PARENT_LATENT_SHA        = "503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a"
  NOISE_SELECTION_PATH     = "/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/render-v1-noise-selection.json"
  PREFIX_KERNEL            = "hy_nomas_soft_route_copy_prefix_zero_f32"
  PREFIX_SOURCE            = ML::Video::MetalOps::SOURCE + <<-METAL
    kernel void hy_nomas_soft_route_copy_prefix_zero_f32(
        device const float *source [[buffer(0)]],
        device float *destination [[buffer(1)]],
        constant uint &prefix_count [[buffer(2)]],
        constant uint &total_count [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
      if (index < total_count) destination[index] = index < prefix_count ? source[index] : 0.0f;
    }
    METAL
  CONFIG = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: VIDEO,
    text_tokens: TEXT, dim: DIM, heads: HEADS, head_dim: HEAD_DIM, ffn_dim: 8192,
    attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
  CACHE_CONFIG = HyNomasDiTRouteFactorialGPU::CACHE_CONFIG

  def sha256_file(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      buffer = Bytes.new(8 * 1024 * 1024)
      while (count = io.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.hexfinal
  end

  def exact_differences(left : Array(Float32), right : Array(Float32)) : Int64
    raise ArgumentError.new("comparison tensor extent mismatch") unless left.size == right.size
    left.each_with_index.count { |value, index| value.unsafe_as(UInt32) != right[index].unsafe_as(UInt32) }.to_i64
  end

  # The renderer encodes the staged weights once per outer Euler step, then
  # reuses that layer for T0 and all three T1 arms. Exercise the same cursor
  # lifecycle without Metal so admission catches a cap/order regression.
  def validate_stack_cursor_lifecycle! : NamedTuple(
    weight_stage_passes: Int32,
    logical_full_forward_calls: Int32,
    blocks_per_pass: Int32,
    fifth_pass_rejected: Bool,
  )
    cursor = ML::Video::HyNomasFullPassCursor.new(BLOCKS,
      WEIGHT_STAGE_PASSES_PER_CHAIN)
    chain_id = 0x48594e4f4d4153_u64
    WEIGHT_STAGE_PASSES_PER_CHAIN.times do |pass|
      BLOCKS.times do |layer|
        cursor.validate_block!(chain_id, layer.to_i32,
          previous_terminal: false, previous_wait_count: 0)
        cursor.finish_block!(chain_id, layer.to_i32)
      end
      unless cursor.forwards_on_chain == pass + 1 && cursor.next_block == BLOCKS
        raise ArgumentError.new("Hy-Nomas render cursor did not finish the expected ordered layer pass")
      end
    end

    fifth_pass_rejected = false
    begin
      cursor.validate_block!(chain_id, 0,
        previous_terminal: false, previous_wait_count: 0)
    rescue ex : ArgumentError
      fifth_pass_rejected = ex.message == "Hy-Nomas stack reached its per-chain forward limit"
    end
    unless fifth_pass_rejected
      raise ArgumentError.new("Hy-Nomas render cursor accepted a fifth weight-stage pass")
    end
    {
      weight_stage_passes:        cursor.forwards_on_chain,
      logical_full_forward_calls: LOGICAL_FULL_FORWARD_CALLS,
      blocks_per_pass:            BLOCKS,
      fifth_pass_rejected:        fifth_pass_rejected,
    }
  end

  def read_f32(path : String, sha : String, count : Int32) : Array(Float32)
    raise ArgumentError.new("accepted final target must be a regular absolute file") unless Path[path].absolute? && File.file?(path) && !File.symlink?(path)
    raise ArgumentError.new("accepted final target byte count mismatch") unless File.size(path) == count.to_i64 * 4
    raise ArgumentError.new("accepted final target digest mismatch") unless sha256_file(path) == sha
    raw = File.read(path).to_slice
    Array(Float32).new(count) { |index| IO::ByteFormat::LittleEndian.decode(Float32, raw[index * 4, 4]) }
  end

  def bindings(admission : Admission) : Hash(String, String)
    {
      "parent_sha256"          => admission.parent.sha256,
      "bias_sha256"            => admission.bias.sha256,
      "roi_source_sha256"      => ROI_SOURCE_SHA,
      "noise_selection_sha256" => admission.noise_sha256,
      "initial_latent_sha256"  => PARENT_LATENT_SHA,
      "t1_conditioning_sha256" => admission.anchor.file.sha256,
      "t1_qwen_sha256"         => QWEN_SHA,
    }
  end

  class Admission
    getter parent : HyNomasDiTRouteFactorialProbe::Reference
    getter anchor : HyNomasGeneration::Inputs
    getter bias : HyNomasSoftRouteProbe::Biases
    getter parent_path : String
    getter parent_sha256 : String
    getter anchor_path : String
    getter anchor_sha256 : String
    getter noise_path : String
    getter noise_sha256 : String
    getter noise_document : JSON::Any
    getter bias_path : String
    getter bias_sha256 : String
    getter anchor_payload_path : String
    getter anchor_payload_sha256 : String

    def initialize(parent_path : String, parent_sha : String,
                   anchor_path : String, anchor_sha : String,
                   bias_path : String, bias_sha : String,
                   noise_path : String, noise_sha : String)
      @parent = HyNomasDiTRouteFactorialProbe::Reference.new(parent_path, parent_sha)
      raise ArgumentError.new("render parent digest differs from the frozen direct-route reference") unless @parent.sha256 == PARENT_SHA
      @parent_path = @parent.path
      @parent_sha256 = @parent.sha256
      raise ArgumentError.new("render anchor digest differs from the frozen conditioning crop") unless anchor_sha == ANCHOR_SHA
      @anchor_path = File.realpath(anchor_path)
      @anchor_sha256 = anchor_sha
      @anchor = HyNomasGeneration::Inputs.new(@anchor_path, anchor_sha, conditioned: true)
      @anchor_payload_path = @anchor.file.payload
      @anchor_payload_sha256 = @anchor.file.document["payload_sha256"].as_s
      raise ArgumentError.new("render Qwen crop digest differs from the frozen conditioning state") unless @anchor.records["input.qwen_bf16"].sha256 == QWEN_SHA
      raise ArgumentError.new("render bias digest differs from the frozen authored priors") unless bias_sha == BIAS_SHA
      @bias = HyNomasSoftRouteProbe::Biases.new(bias_path, bias_sha)
      @bias_path = @bias.path
      @bias_sha256 = @bias.sha256
      @noise_path = File.realpath(noise_path)
      unless Path[noise_path].absolute? && File.file?(@noise_path) && !File.symlink?(noise_path)
        raise ArgumentError.new("noise-selection proof must be a regular absolute file")
      end
      @noise_sha256 = HyNomasSoftRouteRender.sha256_file(@noise_path)
      unless @noise_sha256 == noise_sha && noise_sha == NOISE_SELECTION_SHA
        raise ArgumentError.new("noise-selection proof digest mismatch")
      end
      @noise_document = JSON.parse(File.read(@noise_path))
      validate_noise_selection!
      unless @anchor.records["input.latent"].sha256 == ANCHOR_LATENT_SHA &&
             @parent.inputs["t1.initial_latent"].size == LATENT
        raise ArgumentError.new("render parent or explicitly unused anchor-noise identity mismatch")
      end
      roi = @parent.document["source_inputs"]["roi"]
      unless roi["manifest_sha256"].as_s == ROI_SOURCE_SHA &&
             roi["payload_sha256"].as_s == HyNomasDiTRouteFactorialProbe::ROI_PAYLOAD_SHA256
        raise ArgumentError.new("render accepted ROI source binding mismatch")
      end
      unless @bias.values["zero"].all?(&.zero?) &&
             @bias.values["A1_high"].any? { |value| value > 0 } &&
             @bias.values["A2_high"].any? { |value| value > 0 }
        raise ArgumentError.new("render bias cases do not match the admitted authored priors")
      end
      target = HyNomasSoftRouteRender.read_f32(ACCEPTED_FINAL_PATH, ACCEPTED_FINAL_SHA, LATENT)
      raise ArgumentError.new("accepted final latent is not finite") unless target.all?(&.finite?)
    end

    def close : Nil
      # TensorFile keeps no persistent descriptor; this is an ownership hook
      # matching Runner's single-use lifecycle.
    end

    private def validate_noise_selection! : Nil
      doc = @noise_document
      unless doc["schema"].as_s == "cogni_ml.hy_nomas_soft_route_render_noise_selection/v1" &&
             doc["decision"]["anchor_input_latent_used"].as_bool == false &&
             doc["decision"]["selected_record"].as_s == "t1.initial_latent" &&
             doc["anchor"]["manifest_sha256"].as_s == @anchor_sha256 &&
             doc["anchor"]["sha256"].as_s == ANCHOR_LATENT_SHA &&
             doc["selected_parent"]["source_reference_sha256"].as_s == @parent.sha256 &&
             doc["selected_parent"]["accepted_source_range_sha256"].as_s == PARENT_LATENT_SHA &&
             doc["selected_parent"]["matches_accepted_source_record"].as_bool
        raise ArgumentError.new("noise-selection proof does not bind the parent latent and unused anchor latent")
      end
    end
  end

  class Attention < ML::Video::HyNomasMPSHistoryProvider
    getter cache_key : ML::MetalBuffer
    getter cache_value : ML::MetalBuffer
    @arena : ML::Video::Arena
    @workspaces = [] of ML::Video::HyNomasHistoryAttentionWorkspace
    @plans = {} of String => ML::Video::HyNomasHistoryAttentionMPSPlan
    getter case_calls : Hash(String, Int32)
    @selected = "T0"
    @case_calls = {"T0" => 0, "zero" => 0, "A1_high" => 0, "A2_high" => 0}
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@query : ML::MetalBuffer, @key : ML::MetalBuffer,
                   @value : ML::MetalBuffer, @output : ML::MetalBuffer,
                   parent : HyNomasDiTRouteFactorialProbe::Reference,
                   biases : Hash(String, Array(Float32)))
      @arena = ML::Video::Arena.new
      @cache_key = @arena.alloc("render.T0.key", HISTORY * DIM)
      @cache_value = @arena.alloc("render.T0.value", HISTORY * DIM)
      begin
        t0 = ML::Video::HyNomasHistoryAttentionWorkspace.new(0, HISTORY,
          HEADS, HEAD_DIM, source_bf16_attention: true)
        @workspaces << t0
        t0.freeze_history_inventory!
        @plans["T0"] = ML::Video::HyNomasHistoryAttentionMPSPlan.new(t0, @query, @key, @value, @output)
        CASES.each do |name|
          bias = name == "zero" ? nil : biases[name]
          work = ML::Video::HyNomasHistoryAttentionWorkspace.new(HISTORY, QUERY,
            HEADS, HEAD_DIM, query_labels: parent.native_query_labels_for("M11"),
            key_labels: parent.key_labels, source_bf16_attention: true,
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
      raise ArgumentError.new("released soft-route attention") if @released
      raise ArgumentError.new("unknown soft-route attention arm") unless @plans.has_key?(name)
      @selected = name
    end

    def resident_bytes : Int64
      @arena.resident_bytes + @workspaces.sum(&.resident_bytes)
    end

    def validate_bindings!(tokens : Int32, heads : Int32, head_dim : Int32,
                           query : ML::MetalBuffer, key : ML::MetalBuffer,
                           value : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      unless tokens == HISTORY && heads == HEADS && head_dim == HEAD_DIM &&
             query.same?(@query) && key.same?(@key) && value.same?(@value) && output.same?(@output)
        raise ArgumentError.new("soft-route attention bindings changed")
      end
      @plans.each_value { |plan| plan.validate_bindings!(query, key, value, output) }
    end

    def validate_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("released soft-route attention") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("soft-route attention cannot overlap command chains")
        end
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain, block_index : Int32) : Nil
      raise ArgumentError.new("render reuses registered block zero") unless block_index == 0
      validate_chain!(chain)
      @last_chain = chain
      @case_calls[@selected] += 1
      @plans[@selected].encode(chain, @selected == "T0" ? -1 : 0)
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("drain attention before release") unless chain.completed?
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
    getter plan : Hash(String, Int64)
    @admission : Admission
    @budget : Int64
    @inputs : ML::Video::Arena
    @states : ML::Video::Arena
    @snapshots : ML::Video::Arena
    @stage : ML::Video::Arena
    @globals : ML::Video::Arena
    @loader : ML::Video::HyNomasBF16Stack
    @global_file : ML::Video::CheckpointFile
    @global_plan : ML::Video::WeightPlan
    @global_prepared : ML::Video::PreparedWeights
    @serial : ML::Video::HyNomasMPSSerialWorkspace? = nil
    @block : ML::Video::HyNomasDualStreamBlock? = nil
    @workspace : ML::Video::HyNomasMPSWorkspace? = nil
    @attention : Attention? = nil
    @refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
    @patches : Hash(String, ML::Video::HyNomasPatchMPSPlan)
    @final : ML::Video::HyNomasFinalMPSPlan? = nil
    @eulers : Hash(String, ML::Video::HyNomasEulerPlan)
    @prefix_pipeline : ML::Metal::ComputePipeline? = nil
    @chain : ML::Metal::MPSCommandChain? = nil
    @wait_attempted = false
    @released = false

    def self.memory_plan(parent : HyNomasDiTRouteFactorialProbe::Reference,
                         anchor : HyNomasGeneration::Inputs,
                         biases : Hash(String, Array(Float32))) : Hash(String, Int64)
      parts = HyNomasDiTRouteFactorialGPU.memory_plan(biases)
      extra_attention = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(HISTORY, QUERY,
        HEADS, HEAD_DIM, query_labels: parent.native_query_labels_for("M11"),
        key_labels: parent.key_labels, source_bf16_attention: true,
        action_logit_bias: biases["A2_high"]).workspace_bytes.to_i64
      parts["attention_workspaces"] += extra_attention
      global_bytes = HyNomasQ4DiTIO::GLOBAL_BINDINGS.sum(0_i64) do |binding|
        binding.shape.reduce(1_i64) { |n, d| n * d } * 2
      end
      parts["all_global_weights"] = global_bytes - parts["final_weights"]
      parts["refiner"] = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(anchor.mask, retain_padding: true).total
      parts["patch_plans"] = 3_i64 * (VIDEO.to_i64 * 32 + VIDEO.to_i64 * DIM + DIM.to_i64 * 32) * 4
      parts["qwen_input"] = 1_280_i64 * 3_584 * 2
      parts["render_latents"] = 3_i64 * LATENT * 4
      parts["euler_schedules"] = 3_i64 * OUTER_STEPS * 2 * sizeof(Float32)
      transient = parts["setup_transient_attention_bytes"]
      parts.delete("aggregate_named_bytes")
      parts.delete("peak_named_bytes")
      parts["aggregate_named_bytes"] = parts.values.sum(0_i64) - transient
      parts["peak_named_bytes"] = parts["aggregate_named_bytes"] + transient
      parts
    end

    # HyNomasBF16Stack accounts for all 54 resident BF16 controls, one shared
    # BF16 stage, and the remaining named plan as reserve. The Q4 admission's
    # payload_bytes is compressed disk residency and must never stand in for
    # the BF16 stack's resident control bytes.
    def self.bf16_stack_reserve_bytes(plan : Hash(String, Int64), stage_bytes : Int64,
                                      resident_bytes : Int64) : Int64
      peak = plan["peak_named_bytes"]? || raise ArgumentError.new("memory plan has no named peak")
      expected_stage = plan["block_weight_stage"]? || raise ArgumentError.new("memory plan has no block stage")
      expected_resident = plan["resident_block_weights"]? || raise ArgumentError.new("memory plan has no resident BF16 weight total")
      unless peak > 0 && expected_stage > 0 && expected_resident > 0 &&
             stage_bytes == expected_stage && resident_bytes == expected_resident
        raise ArgumentError.new("BF16 stack accounting does not match the exact named plan")
      end
      if resident_bytes > peak || stage_bytes > peak - resident_bytes
        raise ArgumentError.new("BF16 resident controls and stage exceed the named peak")
      end
      reserve = peak - resident_bytes - stage_bytes
      unless reserve >= 0 && resident_bytes + stage_bytes + reserve == peak
        raise ArgumentError.new("BF16 stack reservation does not reconcile to the named peak")
      end
      reserve
    end

    # CPU-only admission uses the real authenticated Q4 payload as a negative
    # control: substituting its compressed byte count for resident BF16 bytes
    # must fail the same exact contract used by Runner.initialize.
    def self.validate_bf16_stack_budget!(plan : Hash(String, Int64), stage_bytes : Int64,
                                         q4_payload_bytes : Int64) : NamedTuple(
      bf16_resident_bytes: Int64,
      stage_bytes: Int64,
      reserve_bytes: Int64,
      admitted_bytes: Int64,
      q4_payload_bytes: Int64,
      q4_substitution_rejected: Bool,
    )
      peak = plan["peak_named_bytes"]? || raise ArgumentError.new("memory plan has no named peak")
      resident = plan["resident_block_weights"]? || raise ArgumentError.new("memory plan has no resident BF16 weight total")
      reserve = bf16_stack_reserve_bytes(plan, stage_bytes, resident)
      admitted = resident + stage_bytes + reserve
      raise ArgumentError.new("BF16 stack admitted bytes differ from the named peak") unless admitted == peak
      raise ArgumentError.new("Q4 payload byte count is invalid") unless q4_payload_bytes > 0 && q4_payload_bytes != resident
      substitution_rejected = false
      begin
        bf16_stack_reserve_bytes(plan, stage_bytes, q4_payload_bytes)
      rescue ArgumentError
        substitution_rejected = true
      end
      unless substitution_rejected
        raise ArgumentError.new("compressed Q4 payload was accepted as BF16 resident bytes")
      end
      {
        bf16_resident_bytes:      resident,
        stage_bytes:              stage_bytes,
        reserve_bytes:            reserve,
        admitted_bytes:           admitted,
        q4_payload_bytes:         q4_payload_bytes,
        q4_substitution_rejected: substitution_rejected,
      }
    end

    def initialize(@admission : Admission, @budget : Int64)
      validate_stack_cursor_lifecycle!
      parent = @admission.parent
      anchor = @admission.anchor
      biases = {"zero"    => @admission.bias.values["zero"],
                "A1_high" => @admission.bias.values["A1_high"],
                "A2_high" => @admission.bias.values["A2_high"]}
      @plan = self.class.memory_plan(parent, anchor, biases)
      raise ArgumentError.new("four-step named-buffer plan exceeds budget") unless @budget > 0 && @plan["peak_named_bytes"] <= @budget
      cache_admission = ML::Video::HyNomasQ4StackIndex::Admission.new(parent.cache_index_path,
        parent.cache_index_sha256, UPSTREAM_SHA, CACHE_CONFIG)
      @global_file = ML::Video::BF16WeightCache.authenticate(cache_admission.globals_bf16_path,
        cache_admission.globals_bf16_sha256)
      bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |binding|
        ML::Video::WeightBinding.new(binding.target, binding.source, binding.shape,
          ML::GGUF::SafeTensorDType::BF16)
      end
      @global_plan = ML::Video::WeightPlan.new([@global_file], bindings, @budget,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      unless @global_plan.destination_bytes == @plan["all_global_weights"] + @plan["final_weights"] &&
             @global_plan.staging_bytes == 0
        raise ArgumentError.new("global BF16 plan does not match admitted bindings")
      end
      stage_bytes = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG).sum(0_i64) do |binding|
        binding.shape.reduce(1_i64) { |n, d| n * d } * 2
      end
      stack_budget = self.class.validate_bf16_stack_budget!(@plan, stage_bytes,
        cache_admission.payload_bytes)
      reserve = stack_budget[:reserve_bytes]
      @loader = ML::Video::HyNomasBF16Stack.new(cache_admission,
        parent.cache_index_sha256, UPSTREAM_SHA, CACHE_CONFIG, @budget,
        stage_bytes: stage_bytes, reserve_bytes: reserve,
        max_forwards_per_chain: WEIGHT_STAGE_PASSES_PER_CHAIN)

      @inputs = ML::Video::Arena.new
      @states = ML::Video::Arena.new
      @snapshots = ML::Video::Arena.new
      @stage = ML::Video::Arena.new
      @globals = ML::Video::Arena.new
      @patches = {} of String => ML::Video::HyNomasPatchMPSPlan
      @eulers = {} of String => ML::Video::HyNomasEulerPlan
      @global_prepared = ML::Video::PreparedWeights.new(@global_plan, @globals)
      cleanup = [-> { @global_prepared.release }]
      begin
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG).each do |binding|
          @stage.alloc_weight(binding.target,
            binding.shape.reduce(1_i64) { |n, d| n * d }.to_i32,
            ML::Video::WeightDType::BF16)
        end
        @loader.load!(@stage)
        img = @inputs.alloc("working.image", IMAGE)
        txt = @inputs.alloc("working.text", TEXT_STORAGE)
        vec = @inputs.alloc("working.vec", DIM)
        cos = @inputs.alloc("working.cos", ROPE)
        sin = @inputs.alloc("working.sin", ROPE)
        qwen = anchor.upload!("input.qwen_bf16", @inputs)
        @inputs.upload("T0.vector", parent.inputs["t0.vector"])
        {"cos" => parent.rope_cos, "sin" => parent.rope_sin}.each do |kind, values|
          raise ArgumentError.new("parent RoPE bank count mismatch") unless values.size == 2 * ROPE
          @inputs.upload("T0.#{kind}", values[0, ROPE])
          @inputs.upload("T1.#{kind}", values[ROPE, ROPE])
        end
        padded_t0 = Array(Float32).new(TEXT_STORAGE, 0.0_f32)
        parent.inputs["t0.text_valid"].each_with_index { |value, index| padded_t0[index] = value }
        @inputs.upload("T0.image", parent.inputs["t0.patch"])
        @inputs.upload("T0.text", padded_t0)
        @states.alloc("T0.image", IMAGE)
        @states.alloc("T0.text", TEXT_STORAGE)
        CASES.each do |name|
          latent = @states.upload("#{name}.latent", parent.inputs["t1.initial_latent"])
          @states.alloc("#{name}.image", IMAGE)
          @states.alloc("#{name}.text", TEXT_STORAGE)
          @patches[name] = ML::Video::HyNomasPatchMPSPlan.new(latent, VIDEO,
            @globals, source_bf16: true)
        end
        @refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, anchor.mask,
          1_000.0_f32, @globals,
          ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(anchor.mask, retain_padding: true).total,
          retain_padding: true, prelude_input_bf16_control: true,
          prelude_linear_bf16_control: true, prelude_silu_bf16_control: true,
          prelude_second_linear_bf16_control: true, prelude_add_bf16_control: true,
          block_boundaries_bf16_control: true, qkv_bf16_control: true,
          source_attention_control: true, source_projection_residual_control: true,
          source_norm2_control: true, source_fc1_control: true,
          source_mlp_silu_control: true, source_fc2_residual_control: true,
          source_mean_autocast: true)
        cleanup << -> { refiner.release }
        CASES.each do |name|
          @inputs.alloc("#{name}.velocity", LATENT)
          @eulers[name] = ML::Video::HyNomasEulerPlan.new(@states["#{name}.latent"],
            @inputs["#{name}.velocity"], refiner.checkpoints["timestep"],
            steps: OUTER_STEPS, shift: 5.0_f32)
        end
        @serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG, [@stage, @stage],
          img, txt, vec, vec, cos, sin, attention_query_chunk: 256,
          attention_heads_per_group: HEADS, valid_text_tokens: T0_VALID,
          allow_shared_modulation: true,
          workspace_factory: ML::Video::HyNomasSourceBF16MPSWorkspace.factory)
        cleanup << -> { serial.release }
        @block = serial.blocks.first
        @workspace = serial.soft_route_render_workspace
        joint = block.checkpoints
        @attention = Attention.new(joint["joint.query"], joint["joint.key"],
          joint["joint.value"], joint["joint.attention"], parent, biases)
        begin
          serial.attach_history_provider!(attention)
        rescue ex
          attention.release
          raise ex
        end
        @final = ML::Video::HyNomasFinalMPSPlan.new(img, refiner.vec, VIDEO,
          @globals, source_bf16: true, source_autocast: true)
        cleanup << -> { final.release }
        CASES.each do |name|
          (1..OUTER_STEPS).each { |step| @snapshots.alloc("#{name}.step#{step}.latent", LATENT) }
          @snapshots.alloc("#{name}.step1.velocity", LATENT)
        end
        @snapshots.alloc("zero.step1.patch", IMAGE)
        @snapshots.alloc("zero.step1.text_valid", VALID_TEXT)
        @snapshots.alloc("zero.step1.vector", DIM)
        @prefix_pipeline = ML::Metal::PipelineCache.get(PREFIX_KERNEL) do
          ML::Metal::ComputePipeline.new(PREFIX_KERNEL, PREFIX_SOURCE)
        end
        actual = @loader.cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes +
                 @inputs.resident_bytes + @states.resident_bytes + @snapshots.resident_bytes +
                 serial.resident_bytes + refiner.resident_bytes +
                 @patches.values.sum(&.resident_bytes) + @eulers.values.sum(&.resident_bytes) +
                 final.resident_bytes
        raise ArgumentError.new("actual named render allocations exceed planned peak") if actual > @plan["peak_named_bytes"] || actual > @budget
        @plan["actual_setup_named_bytes"] = actual
      rescue ex
        cleanup.reverse_each(&.call)
        @patches.each_value(&.release)
        @eulers.each_value(&.release)
        @loader.release
        [@inputs, @states, @snapshots, @stage, @globals].each(&.release)
        @global_file.close
        @admission.close
        @released = true
        raise ex
      end
    end

    def run : NamedTuple(outputs: Hash(String, Array(Float32)), terminal_waits: Int32,
      encode_allocations: Int64, encode_readbacks: Int64, seconds: Float64,
      case_calls: Hash(String, Int32))
      raise ArgumentError.new("render is single-use") if @chain || @released
      chain = ML::Metal::MPSCommandChain.new
      @chain = chain
      before_alloc = ML::MetalBuffer.hy_text_probe_allocations
      before_reads = ML::MetalBuffer.hy_text_probe_readbacks
      started = Time.instant
      @global_prepared.encode(chain.encoder)
      OUTER_STEPS.times do |step|
        copy(chain, @inputs["T0.image"], @states["T0.image"], IMAGE)
        copy(chain, @inputs["T0.text"], @states["T0.text"], TEXT_STORAGE)
        CASES.each { |name| @eulers[name].encode_timestep(chain, step) }
        refiner.encode(chain)
        CASES.each do |name|
          @patches[name].encode(chain)
          copy(chain, @patches[name].output, @states["#{name}.image"], IMAGE)
        end
        copy_prefix_zero(chain, refiner.text_output, @states["zero.text"])
        CASES[1..].each { |name| copy(chain, @states["zero.text"], @states["#{name}.text"], TEXT_STORAGE) }
        if step == 0
          copy(chain, @patches["zero"].output, @snapshots["zero.step1.patch"], IMAGE)
          copy(chain, refiner.text_output, @snapshots["zero.step1.text_valid"], VALID_TEXT)
          copy(chain, refiner.vec, @snapshots["zero.step1.vector"], DIM)
        end
        BLOCKS.times do |layer|
          @loader.encode!(chain, layer, @stage)
          encode_state(chain, "T0", "T0")
          joint = block.checkpoints
          copy(chain, joint["joint.key"], attention.cache_key, HISTORY * DIM)
          copy(chain, joint["joint.value"], attention.cache_value, HISTORY * DIM)
          CASES.each { |name| encode_state(chain, name, name == "T0" ? "T0" : "T1") }
        end
        CASES.each do |name|
          copy(chain, @states["#{name}.image"], @inputs["working.image"], IMAGE)
          final.encode(chain)
          copy(chain, final.output, @inputs["#{name}.velocity"], LATENT)
          if step == 0
            copy(chain, final.output, @snapshots["#{name}.step1.velocity"], LATENT)
          end
          @eulers[name].encode_update(chain, step)
          copy(chain, @states["#{name}.latent"], @snapshots["#{name}.step#{step + 1}.latent"], LATENT)
        end
      end
      allocations = ML::MetalBuffer.hy_text_probe_allocations - before_alloc
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
      raise "named Metal allocation occurred during render encode" unless allocations == 0
      raise "activation readback occurred before terminal wait" unless readbacks == 0
      raise "intermediate command wait occurred during render encode" unless chain.wait_count == 0
      expected_calls = {"T0" => BLOCKS * OUTER_STEPS, "zero" => BLOCKS * OUTER_STEPS,
                        "A1_high" => BLOCKS * OUTER_STEPS, "A2_high" => BLOCKS * OUTER_STEPS}
      raise "render attention case-call inventory mismatch" unless attention.case_calls == expected_calls
      @wait_attempted = true
      chain.commit_and_wait
      raise "render requires exactly one terminal wait" unless chain.completed? && chain.wait_count == 1
      outputs = {} of String => Array(Float32)
      CASES.each do |name|
        (1..OUTER_STEPS).each do |step|
          key = "#{name}.step#{step}.latent"
          outputs[key] = @snapshots[key].read(LATENT)
        end
        key = "#{name}.step1.velocity"
        outputs[key] = @snapshots[key].read(LATENT)
      end
      ["zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector"].each do |key|
        size = key == "zero.step1.patch" ? IMAGE : (key.ends_with?("text_valid") ? VALID_TEXT : DIM)
        outputs[key] = @snapshots[key].read(size)
      end
      {outputs: outputs, terminal_waits: chain.wait_count,
       encode_allocations: allocations, encode_readbacks: readbacks,
       seconds: (Time.instant - started).total_seconds,
       case_calls: attention.case_calls.dup}
    ensure
      if active = @chain
        unless active.completed? || @wait_attempted
          @wait_attempted = true
          active.commit_and_wait
        end
      end
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("drain render before release") unless chain.completed?
      end
      @patches.each_value(&.release)
      @eulers.each_value(&.release)
      final.release
      serial.release
      refiner.release
      @loader.release
      @global_prepared.release
      [@inputs, @states, @snapshots, @stage, @globals].each(&.release)
      @global_file.close
      @chain.try(&.release)
      @admission.close
      @released = true
    end

    private def copy(chain : ML::Metal::MPSCommandChain, source : ML::MetalBuffer,
                     destination : ML::MetalBuffer, count : Int32) : Nil
      ML::Video::MetalOps.copy(chain.encoder, source, destination, count)
    end

    private def copy_prefix_zero(chain : ML::Metal::MPSCommandChain,
                                 source : ML::MetalBuffer, destination : ML::MetalBuffer) : Nil
      chain.encoder.set_pipeline(prefix_pipeline).set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(VALID_TEXT.to_u32, 2).set_value(TEXT_STORAGE.to_u32, 3)
        .dispatch_1d(TEXT_STORAGE)
    end

    private def encode_state(chain : ML::Metal::MPSCommandChain, name : String,
                             phase : String) : Nil
      image = @inputs["working.image"]
      text = @inputs["working.text"]
      vec = @inputs["working.vec"]
      cos = @inputs["working.cos"]
      sin = @inputs["working.sin"]
      copy(chain, @states["#{name}.image"], image, IMAGE)
      copy(chain, @states["#{name}.text"], text, TEXT_STORAGE)
      if name == "T0"
        copy(chain, @inputs["T0.vector"], vec, DIM)
      else
        copy(chain, refiner.vec, vec, DIM)
      end
      copy(chain, @inputs["#{phase}.cos"], cos, ROPE)
      copy(chain, @inputs["#{phase}.sin"], sin, ROPE)
      attention.select!(name)
      block.encode_mps(chain, image, text, vec, vec, cos, sin, workspace)
      copy(chain, block.image_output, @states["#{name}.image"], IMAGE)
      copy(chain, block.text_output, @states["#{name}.text"], TEXT_STORAGE)
    end

    private def refiner : ML::Video::HyNomasTextRefinerMPSPlan
      @refiner || raise ArgumentError.new("text refiner is unavailable")
    end

    private def serial : ML::Video::HyNomasMPSSerialWorkspace
      @serial || raise ArgumentError.new("serial workspace is unavailable")
    end

    private def block : ML::Video::HyNomasDualStreamBlock
      @block || raise ArgumentError.new("DiT block is unavailable")
    end

    private def workspace : ML::Video::HyNomasMPSWorkspace
      @workspace || raise ArgumentError.new("DiT block workspace is unavailable")
    end

    private def attention : Attention
      @attention || raise ArgumentError.new("soft-route attention is unavailable")
    end

    private def final : ML::Video::HyNomasFinalMPSPlan
      @final || raise ArgumentError.new("final head is unavailable")
    end

    private def prefix_pipeline : ML::Metal::ComputePipeline
      @prefix_pipeline || raise ArgumentError.new("prefix copy pipeline is unavailable")
    end
  end
end

class ML::Video::HyNomasMPSSerialWorkspace
  def soft_route_render_workspace : ML::Video::HyNomasMPSWorkspace
    workspace
  end
end
