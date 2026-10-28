require "json"
require "digest/sha256"
require "./hy_nomas_probe_instrumentation"
require "./hy_nomas_q4_dit_io"
require "./pixelumm_two_block_probe"
require "../../src/ml/video/hy_nomas_bf16_stack"
require "../../src/ml/video/hy_nomas_euler"
require "../../src/ml/video/pixelumm_generation"
require "../../src/ml/video/pixelumm_decoder_head_mps"

# Opt-in, bounded first segment. All neural handoffs stay on GPU. DiT and
# PixelUMM use separate drained chains so their weights need not coexist.
module HyNomasGeneration
  extend self
  GIB    = 1024_i64**3
  CONFIG = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 4320,
    text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)
  MASK            = Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 }
  INDEX           = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/q4-cache-v2/index.json"
  INDEX_SHA       = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
  PIXEL_CACHE     = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027/source/pixelumm-thirty-six-bf16-cache.json"
  PIXEL_CACHE_SHA = "74ee5e84f2b63af0189ffd7be0e4c758b8dfa9ff925e3c9cb389d207ce546dba"
  STEM_FILE       = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.safetensors"
  STEM_SHA        = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
  HEAD_FILE       = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-head-20261027/source/head-f32.safetensors"
  HEAD_SHA        = "d0e985e45b6b5fcebde381dd4b61f3b77394bd288323cd8afe2a7604584e6e9f"
  # This entry point is one fixed, producer-authenticated first-segment pilot.
  # Pinning its whole manifest also binds the prompt, saved Qwen crop, source
  # identities, seeds and geometry; a caller-selected digest cannot change it.
  INPUTS_SHA = "5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b"
  alias TensorFile = PixelUMMTwoBlockProbe::TensorFile
  alias Record = PixelUMMBlockProbe::Record

  enum DiTWeights
    BF16
    Q4K
  end

  def parse_dit_weights(value : String) : DiTWeights
    case value
    when "bf16" then DiTWeights::BF16
    when "q4-k" then DiTWeights::Q4K
    else             raise ArgumentError.new("DiT weights must be bf16 or q4-k")
    end
  end

  def dit_weights_name(mode : DiTWeights) : String
    mode == DiTWeights::BF16 ? "bf16" : "q4-k"
  end

  # Both loaders keep the same fixed BF16 block stage. Only the all-block
  # residents differ; Q4 payloads include their exact BF16 vector exceptions.
  def dit_resident_weight_bytes(mode : DiTWeights, stage : Int64, q4_payload : Int64) : Int64
    raise ArgumentError.new("DiT payload and stage bytes must be positive") unless stage > 0 && q4_payload > 0
    payload = mode == DiTWeights::BF16 ? 54_i64 * stage : q4_payload
    payload + stage
  end

  # Crystal's string modes do not support Python's "x" flag. Preserve the
  # no-replacement contract with the same exclusive descriptor API as probes.
  def write_fresh_output(path : String, & : IO::FileDescriptor ->) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("preview output must be fresh and writable: #{path}") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      yield io
    ensure
      io.close
    end
  end

  def input_shapes : Hash(String, Tuple(String, Array(Int32)))
    {
      "input.latent"     => {"F32", [1, 32, 30, 9, 16]},
      "input.qwen_bf16"  => {"BF16", [1, 1280, 3584]},
      "input.rope_cos"   => {"F32", [4320, 128]},
      "input.rope_sin"   => {"F32", [4320, 128]},
      "decoder.noise"    => {"F32", [1, 3, 120, 288, 512]},
      "decoder.rope_cos" => {"F32", [1, 5760, 1, 128]},
      "decoder.rope_sin" => {"F32", [1, 5760, 1, 128]},
    }
  end

  class Inputs
    getter file : TensorFile
    getter records : Hash(String, Record)
    getter prompt : String

    def initialize(path : String, sha : String)
      HyNomasGeneration.validate_input_identity!(sha)
      @file = TensorFile.new(File.realpath(path), sha)
      doc = @file.document
      unless doc["schema"].as_s == "cogni_ml.hy_nomas_generation_inputs/v1" &&
             doc["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION
        raise ArgumentError.new("generation input provenance mismatch")
      end
      @prompt = doc["prompt"].as_s
      raise ArgumentError.new("generation prompt is empty") if @prompt.empty?
      @records = @file.records("tensors")
      shapes = HyNomasGeneration.input_shapes
      raise ArgumentError.new("generation input inventory mismatch") unless @records.keys.sort == shapes.keys.sort
      shapes.each do |name, (dtype, shape)|
        r = @records[name]
        raise ArgumentError.new("generation input shape/dtype mismatch: #{name}") unless r.dtype == dtype && r.shape == shape
      end
      @file.validate_ranges!(@records.values, contiguous: true)
    end

    def upload!(name : String, arena : ML::Video::Arena) : ML::MetalBuffer
      record = @records[name]
      dtype = record.dtype == "BF16" ? ML::Video::WeightDType::BF16 : ML::Video::WeightDType::F32
      buffer = arena.alloc_weight(name, (record.nbytes // dtype.bytes_per_element).to_i32, dtype)
      @file.upload_into!(record, buffer)
      buffer
    end

    def bytes(names : Array(String)) : Int64
      names.sum { |name| @records[name].nbytes }
    end
  end

  def validate_input_identity!(sha : String) : Nil
    unless sha == INPUTS_SHA
      raise ArgumentError.new("this bounded preview requires the pinned producer-authenticated input manifest")
    end
  end

  # The immutable cache records the geometry of its original qualification
  # fixture. Token extents change activations, but not these parameter tensors.
  # Authenticate against that recorded geometry and separately prove that the
  # runtime consumes exactly the same architecture and weight bindings.
  def weight_cache_config(runtime : ML::Video::HyNomasConfig = CONFIG) : ML::Video::HyNomasConfig
    cache = HyNomasQ4DiTIO::CONFIG
    runtime.validate!
    unless runtime.copy_with(video_tokens: cache.video_tokens) == cache &&
           ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(runtime) ==
             ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(cache)
      raise ArgumentError.new("preview runtime must preserve the authenticated weight-cache architecture")
    end
    cache
  end

  def authenticate_weight_index
    ML::Video::HyNomasQ4StackIndex.load!(File.realpath(INDEX), INDEX_SHA,
      HyNomasQ4DiTIO::SOURCE_SHA, weight_cache_config)
  end

  def stage_bytes : Int64
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG).sum do |b|
      b.shape.reduce(1_i64) { |n, d| n * d } * 2
    end
  end

  def dit_scratch_bytes : Int64
    c = CONFIG.compact_valid_text_prefix(23, 64)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(c).sum { |_, n| n.to_i64 * 4 }
    scratch += 53_i64 * c.joint_rows * c.dim * 4
    attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 4343, 16, 128, 256, 16)
    scratch + ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(c).to_i64 * 4 +
      attention.resident_bytes + 2_i64 * CONFIG.joint_rows * (c.dim + c.heads) * 4 +
      ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(CONFIG, 23, 64)
  end

  def boundary_bytes : Int64
    v = CONFIG.video_tokens.to_i64
    ((v * 32 + v * 2048 + 2048 * 32) +
      (2048 + 2 * v * 2048 + 4096 + 2 * v * 32 + 2_i64 * 2048 * 2048)) * 4
  end

  def dit_reserve_bytes(inputs : Inputs) : Int64
    globals = HyNomasQ4DiTIO::GLOBAL_BINDINGS.sum { |b| b.shape.reduce(1_i64) { |n, d| n * d } * 2 }
    globals + inputs.bytes(%w(input.latent input.qwen_bf16 input.rope_cos input.rope_sin)) +
      boundary_bytes + dit_scratch_bytes +
      ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(MASK, retain_padding: true).total + 8_i64 * 2 * 4
  end

  def stem_bindings : Array(ML::Video::WeightBinding)
    {"embed.weight" => [4096_i64, 3072_i64], "embed.bias" => [4096_i64],
     "cond_proj.weight" => [16384_i64, 32_i64], "cond_proj.bias" => [16384_i64]}.map do |name, shape|
      ML::Video::WeightBinding.new(name, "decoder.#{name}", shape, ML::GGUF::SafeTensorDType::F32)
    end
  end

  def head_bindings : Array(ML::Video::WeightBinding)
    [{"final_norm_weight", "decoder.final_norm.weight", [4096_i64]},
     {"head_norm_weight", "decoder.head.norm.weight", [4096_i64]},
     {"head_linear_weight", "decoder.head.linear.weight", [3072_i64, 4096_i64]},
     {"head_linear_bias", "decoder.head.linear.bias", [3072_i64]}].map do |target, source, shape|
      ML::Video::WeightBinding.new(target, source, shape, ML::GGUF::SafeTensorDType::F32)
    end
  end

  # Pin the complete local subset. Its original range-receipt lineage remains
  # in the source directory; this runtime never downloads or rewrites weights.
  def authenticate_f32(path : String, sha : String) : ML::Video::CheckpointFile
    path = File.realpath(path)
    unless Path[path].absolute? && File.realpath(path) == path && !File.symlink?(path) &&
           PixelUMMBlockProbe.sha_file(path) == sha
      raise ArgumentError.new("generation F32 subset identity mismatch")
    end
    file = ML::Video::CheckpointFile.new(path)
    unless file.tensors.values.all?(&.dtype.f32?)
      file.close
      raise ArgumentError.new("generation source subset requires F32 tensors")
    end
    file
  end

  def emit(event : String, **fields) : Nil
    puts({event: event, monotonic_s: Time.monotonic.total_seconds, pid: Process.pid}.merge(fields).to_json)
    STDOUT.flush
  end

  class DiT
    getter named_bytes : Int64
    getter latent : ML::MetalBuffer
    @inputs_arena : ML::Video::Arena
    @globals : ML::Video::Arena
    @stage : ML::Video::Arena
    @file : ML::Video::CheckpointFile
    @prepared : ML::Video::PreparedWeights
    @loader : ML::Video::HyNomasBF16Stack | ML::Video::HyNomasQ4Stack
    @patch : ML::Video::HyNomasPatchMPSPlan
    @refiner : ML::Video::HyNomasTextRefinerMPSPlan
    @serial : ML::Video::HyNomasMPSSerialWorkspace
    @final : ML::Video::HyNomasFinalMPSPlan
    @euler : ML::Video::HyNomasEulerPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(inputs : Inputs, @latent : ML::MetalBuffer, budget : Int64,
                   mode : DiTWeights = DiTWeights::BF16)
      # Every authenticated file and the exact aggregate are admitted first.
      index_path = File.realpath(INDEX)
      index = HyNomasGeneration.authenticate_weight_index
      @file = ML::Video::BF16WeightCache.authenticate(index.globals_bf16_path, index.globals_bf16_sha256)
      bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |b|
        ML::Video::WeightBinding.new(b.target, b.source, b.shape, ML::GGUF::SafeTensorDType::BF16)
      end
      plan = ML::Video::WeightPlan.new([@file], bindings, budget,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      raise ArgumentError.new("DiT globals must be exact BF16") unless plan.staging_bytes == 0 && plan.entries.size == 48
      @loader = if mode == DiTWeights::BF16
                  ML::Video::HyNomasBF16Stack.new(index_path, INDEX_SHA, HyNomasQ4DiTIO::SOURCE_SHA,
                    HyNomasGeneration.weight_cache_config, budget, stage_bytes: HyNomasGeneration.stage_bytes,
                    reserve_bytes: HyNomasGeneration.dit_reserve_bytes(inputs), max_forwards_per_chain: 8)
                else
                  ML::Video::HyNomasQ4Stack.new(index_path, INDEX_SHA, HyNomasQ4DiTIO::SOURCE_SHA,
                    HyNomasGeneration.weight_cache_config, budget, stage_bytes: HyNomasGeneration.stage_bytes,
                    reserve_bytes: HyNomasGeneration.dit_reserve_bytes(inputs), max_forwards_per_chain: 8)
                end
      @named_bytes = @loader.admitted_bytes
      @inputs_arena = ML::Video::Arena.new
      @globals = ML::Video::Arena.new
      @stage = ML::Video::Arena.new
      @prepared = ML::Video::PreparedWeights.new(plan, @globals)
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG).each do |b|
        @stage.alloc_weight(b.target, b.shape.reduce(1_i64) { |n, d| n * d }.to_i32, ML::Video::WeightDType::BF16)
      end
      HyNomasGeneration.emit("dit_load_blocks", named_bytes: @named_bytes)
      @loader.load!(@stage)
      qwen = inputs.upload!("input.qwen_bf16", @inputs_arena)
      cos = inputs.upload!("input.rope_cos", @inputs_arena)
      sin = inputs.upload!("input.rope_sin", @inputs_arena)
      @patch = ML::Video::HyNomasPatchMPSPlan.new(@latent, 4320, @globals)
      @refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, MASK, 1000_f32, @globals,
        ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(MASK, retain_padding: true).total, retain_padding: true)
      @serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG,
        Array(ML::Video::Arena).new(54) { @stage }, @patch.output, @refiner.text_output,
        @refiner.vec, @refiner.vec_txt, cos, sin, attention_query_chunk: 256,
        attention_heads_per_group: 16, valid_text_tokens: 23, allow_shared_modulation: true,
        full_dit: true, weight_loader: @loader, compact_valid_text: true, compact_text_tokens: 64)
      @final = ML::Video::HyNomasFinalMPSPlan.new(@serial.blocks.last.image_output, @refiner.vec, 4320, @globals)
      @euler = ML::Video::HyNomasEulerPlan.new(@latent, @final.output, @refiner.checkpoints["timestep"], steps: 8, shift: 5_f32)
      actual = @loader.cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes +
               @inputs_arena.resident_bytes + @latent.size + @patch.resident_bytes + @refiner.resident_bytes +
               @serial.resident_bytes + @final.resident_bytes + @euler.resident_bytes
      raise ArgumentError.new("DiT allocation/admission mismatch: #{actual} != #{@named_bytes}") unless actual == @named_bytes
      HyNomasGeneration.emit("dit_prepared", named_bytes: actual)
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("DiT request is single-use") if @released || @chain
      @chain = chain
      @prepared.encode(chain.encoder)
      8.times do |step|
        @euler.encode_timestep(chain, step)
        @patch.encode(chain)
        @refiner.encode(chain)
        @serial.encode(chain)
        @final.encode(chain)
        @euler.encode_update(chain, step)
        HyNomasGeneration.emit("dit_step_encoded", step: step + 1,
          timestep: @euler.timestep_at(step), dt: @euler.dt_at(step))
      end
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("drain DiT before releasing weights") unless chain.completed?
      end
      @euler.release
      @final.release
      @serial.release
      @refiner.release
      @patch.release
      @loader.release
      @prepared.release
      @stage.release
      @globals.release
      @inputs_arena.release
      @file.close
      @released = true
    end
  end

  # CPU-only geometry, identity and aggregate admission for the decoder phase.
  class DecoderMetadata
    getter cache : TensorFile
    getter weights : Hash(String, Record)
    getter stem_file : ML::Video::CheckpointFile
    getter head_file : ML::Video::CheckpointFile
    getter stem_plan : ML::Video::WeightPlan
    getter head_plan : ML::Video::WeightPlan
    getter config = ML::Video::PixelUMMGeneration::Config.new
    getter captures = Array(Array(String)).new(36) { [] of String }
    getter workspace_bytes : Int64
    getter named_bytes : Int64

    def initialize(inputs : Inputs, budget : Int64)
      @cache = TensorFile.new(File.realpath(PIXEL_CACHE), PIXEL_CACHE_SHA, 16_i64 * GIB)
      doc = @cache.document
      unless doc["schema"].as_s == "cogni_ml.pixelumm_bf16_weight_cache/v1" &&
             doc["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             doc["runtime_dtype"].as_s == "BF16"
        raise ArgumentError.new("decoder BF16 cache lineage mismatch")
      end
      @weights = @cache.records("weights")
      PixelUMMTwoBlockProbe.validate_weights!(@weights, 36)
      @cache.validate_ranges!(@weights.values, contiguous: true)
      @stem_file = HyNomasGeneration.authenticate_f32(STEM_FILE, STEM_SHA)
      @head_file = HyNomasGeneration.authenticate_f32(HEAD_FILE, HEAD_SHA)
      @stem_plan = ML::Video::WeightPlan.new([@stem_file], HyNomasGeneration.stem_bindings,
        budget, require_all: true)
      @head_plan = ML::Video::WeightPlan.new([@head_file], HyNomasGeneration.head_bindings,
        budget, resident_dtype: ML::Video::WeightDType::BF16,
        require_all: true, allow_f32_to_bf16: true)
      bc = @config.decoder_config
      @workspace_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(bc, bc.layout,
        @captures, reuse_outputs: true)
      hc = ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 10, height: 18, width: 32)
      @named_bytes = @cache.payload_bytes + @stem_plan.peak_bytes + @head_plan.peak_bytes +
                     inputs.bytes(%w(input.latent decoder.noise decoder.rope_cos decoder.rope_sin)) +
                     @workspace_bytes + ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(hc) +
                     @config.stem_named_bytes_by_name.values.sum + @config.assembly_named_bytes_by_name.values.sum
      raise ArgumentError.new("decoder aggregate named buffers exceed the budget") unless @named_bytes <= budget
    end

    def close : Nil
      @stem_file.close
      @head_file.close
    end
  end

  class Decoder
    getter named_bytes : Int64
    @input_arena : ML::Video::Arena
    @weight_arena : ML::Video::Arena
    @stem_arena : ML::Video::Arena
    @head_arena : ML::Video::Arena
    @stem_prepared : ML::Video::PreparedWeights
    @head_prepared : ML::Video::PreparedWeights
    @stem : ML::Video::PixelUMMGeneration::StemPlan
    @assembly : ML::Video::PixelUMMGeneration::AssemblyPlan
    @workspace : ML::Video::PixelUMMDecoderBlockMPS::Workspace
    @blocks = [] of ML::Video::PixelUMMDecoderBlockMPS::Plan
    @head : ML::Video::PixelUMMDecoderHeadMPS::Plan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(inputs : Inputs, latent : ML::MetalBuffer, metadata : DecoderMetadata)
      @named_bytes = metadata.named_bytes
      # Streaming hashes finish before any decoder Metal allocation.
      metadata.cache.verify_payload!
      @input_arena = ML::Video::Arena.new
      @weight_arena = ML::Video::Arena.new
      @stem_arena = ML::Video::Arena.new
      @head_arena = ML::Video::Arena.new
      @stem_prepared = ML::Video::PreparedWeights.new(metadata.stem_plan, @stem_arena)
      @head_prepared = ML::Video::PreparedWeights.new(metadata.head_plan, @head_arena)
      noise = inputs.upload!("decoder.noise", @input_arena)
      cos = inputs.upload!("decoder.rope_cos", @input_arena)
      sin = inputs.upload!("decoder.rope_sin", @input_arena)
      @stem = ML::Video::PixelUMMGeneration::StemPlan.new(metadata.config, latent: latent, noise: noise,
        weights: ML::Video::PixelUMMGeneration::StemWeights.new(
          embed_weight: @stem_arena["embed.weight"], embed_bias: @stem_arena["embed.bias"],
          cond_weight: @stem_arena["cond_proj.weight"], cond_bias: @stem_arena["cond_proj.bias"]))
      @assembly = ML::Video::PixelUMMGeneration::AssemblyPlan.new(metadata.config)
      config = metadata.config.decoder_config
      @workspace = ML::Video::PixelUMMDecoderBlockMPS::Workspace.new(config,
        captures_by_slot: metadata.captures, max_owned_bytes: metadata.workspace_bytes,
        reuse_outputs: true, max_passes_per_chain: 5)
      input = @stem.output_bf16
      36.times do |slot|
        HyNomasGeneration.emit("decoder_load_block", block: slot)
        weights = {} of String => ML::MetalBuffer
        PixelUMMBlockProbe.weight_shapes.each do |name, _|
          full_name = "block#{slot}.#{name}"
          record = metadata.weights[full_name]
          buffer = @weight_arena.alloc_weight(full_name, (record.nbytes // 2).to_i32, ML::Video::WeightDType::BF16)
          metadata.cache.upload_into!(record, buffer)
          weights[name] = buffer
        end
        block = ML::Video::PixelUMMDecoderBlockMPS::Plan.new(config, input: input, weights: weights,
          cos: cos, sin: sin, max_named_bytes: metadata.workspace_bytes,
          captures: [] of String, workspace: @workspace, slot: slot, ffn_f32: true)
        @blocks << block
        input = block.output
      end
      hc = ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 10, height: 18, width: 32)
      @head = ML::Video::PixelUMMDecoderHeadMPS::Plan.new(hc, input: input,
        weights: @head_arena.buffers.dup, max_named_bytes: ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(hc),
        max_passes_per_chain: 5)
      actual = @input_arena.resident_bytes + latent.size + @weight_arena.resident_bytes +
               @stem_arena.resident_bytes + metadata.stem_plan.staging_bytes +
               @head_arena.resident_bytes + metadata.head_plan.staging_bytes +
               @workspace.named_bytes + @head.named_bytes +
               metadata.config.stem_named_bytes_by_name.values.sum + metadata.config.assembly_named_bytes_by_name.values.sum
      raise ArgumentError.new("decoder allocation/admission mismatch: #{actual} != #{@named_bytes}") unless actual == @named_bytes
      HyNomasGeneration.emit("decoder_prepared", named_bytes: actual)
    end

    def output : ML::MetalBuffer
      @assembly.output_f32
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("decoder request is single-use") if @released || @chain
      @chain = chain
      @stem_prepared.encode(chain.encoder)
      @head_prepared.encode(chain.encoder)
      ML::Video::PixelUMMGeneration::WINDOW_STARTS.each_with_index do |start, index|
        @stem.encode(chain, start)
        @blocks.each(&.encode(chain))
        @head.encode(chain)
        @assembly.encode_merge(chain, window_bf16: @head.output, start_latent_frame: start)
        HyNomasGeneration.emit("decoder_window_encoded", window: index + 1, start_latent_frame: start)
      end
      @assembly.encode_finalize(chain)
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("drain decoder before releasing resources") unless chain.completed?
      end
      @head.release
      @blocks.reverse_each(&.release)
      @workspace.release
      @assembly.release
      @stem.release
      @head_prepared.release
      @stem_prepared.release
      @weight_arena.release
      @head_arena.release
      @stem_arena.release
      @input_arena.release
      @released = true
    end
  end
end
