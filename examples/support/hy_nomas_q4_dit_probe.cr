require "./hy_nomas_q4_dit_io"
require "./hy_nomas_conditioned_eight_probe"
require "../../src/ml/video/hy_nomas_bf16_stack"

# Diagnostic-only accessors keep poison controls out of the production API.
# Inactive RMS rows must be irrelevant even when their scratch contains NaNs.
class ML::Video::HyNomasMPSRMS::Plan
  def hy_dit_probe_scratch : Array(ML::MetalBuffer)
    [square, mean]
  end
end

class ML::Video::HyNomasMPSWorkspace
  def hy_dit_probe_rms_scratch : Array(ML::MetalBuffer)
    @rms_plans.values.flat_map(&.hy_dit_probe_scratch)
  end
end

class ML::Video::HyNomasMPSSerialWorkspace
  def hy_dit_probe_rms_scratch : Array(ML::MetalBuffer)
    workspace.hy_dit_probe_rms_scratch
  end
end

# One camera-off DiT forward. The authenticated upstream Qwen hidden tensor is
# a saved producer boundary, never an expected DiT activation fed back as input.
module HyNomasQ4DiTProbe
  extend self
  CONFIG         = HyNomasQ4DiTIO::CONFIG
  ATOL           =            2e-5
  RTOL           =            2e-5
  SIDECAR_BYTES  = 327_274_496_i64
  STAGE_BYTES    = 302_113_792_i64
  MASK           = Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 }
  CAPTURE_BLOCKS = [0, 7, 53]
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    raw_bit_differences: Int32, max_abs: Float64?, finite: Bool,
    native_sha256: String, first_request_sha256_match: Bool?)

  def shapes : Hash(String, Array(Int32))
    result = {"input.latent" => [1, 32, 6, 22, 40],
              "input.qwen_hidden" => [1, 1280, 3584],
              "input.rope_cos" => [5280, 128], "input.rope_sin" => [5280, 128],
              "img_in" => [1, 5280, 2048],
              "conditioning.projected_text" => [1, 1280, 2048],
              "conditioning.vec" => [1, 2048],
              "final.norm" => [1, 5280, 2048], "final.modulation" => [1, 4096],
              "final.modulated" => [1, 5280, 2048], "final.linear" => [1, 5280, 32],
              "output.latent" => [1, 32, 6, 22, 40]}
    CAPTURE_BLOCKS.each do |i|
      result["block#{i}.img.output"] = [1, 5280, 2048]
      result["block#{i}.txt.output"] = [1, 1280, 2048]
    end
    result
  end

  def validate_document!(doc : JSON::Any, mode : String,
                         index : ML::Video::HyNomasQ4StackIndex::Index) : Nil
    unless {"bf16", "q4"}.includes?(mode) &&
           doc["schema"].as_s == "hy-nomas-q4-dit-reference-v1" &&
           doc["schema_version"].as_i == 1 && doc["precision_mode"].as_s == mode &&
           doc["precision"]["mode"].as_s == mode &&
           doc["tolerance"]["atol"].as_f == ATOL && doc["tolerance"]["rtol"].as_f == RTOL
      raise ArgumentError.new("Full DiT schema, precision mode, or unchanged tolerance mismatch")
    end
    source = doc["source"]
    unless source["checkpoint_sha256"].as_s == HyNomasQ4DiTIO::SOURCE_SHA &&
           source["checkpoint_bytes"].as_i64 == HyNomasQ4DiTIO::SOURCE_BYTES &&
           source["checkpoint_header_sha256"].as_s == HyNomasQ4DiTIO::HEADER_SHA &&
           source["cache_index"]["sha256"].as_s == index.sha256 &&
           source["cache_block_order"].as_a.map(&.as_i) == (0...54).to_a
      raise ArgumentError.new("Full DiT source/checkpoint/index pin mismatch")
    end
    cache_blocks = source["cache_blocks"].as_a
    raise ArgumentError.new("Full DiT must pin all 54 block caches") unless cache_blocks.size == 54
    index.blocks.each_with_index do |pin, i|
      item = cache_blocks[i]
      unless item["block_index"].as_i == i && item["manifest"].as_s == File.basename(pin.manifest_path) &&
             item["manifest_sha256"].as_s == pin.manifest_sha256 && item["payload_sha256"].as_s == pin.payload_sha256 &&
             item["bf16"].as_s == File.basename(pin.bf16_path) && item["bf16_sha256"].as_s == pin.bf16_sha256
        raise ArgumentError.new("Full DiT block cache lineage mismatch: #{i}")
      end
    end
    upstream = doc["upstream"]
    unless upstream["revision"].as_s == HyNomasPretrainedSerialProbe::PINNED_UPSTREAM_REVISION
      raise ArgumentError.new("Full DiT upstream revision mismatch")
    end
    # The fixture producer pins the original source modules, not its own
    # rewritten native implementation. Keep the accepted inventory exact.
    hashes = upstream["source_sha256"].as_h
    expected_hashes = HyNomasConditionedEightProbe::UPSTREAM_SOURCE_SHA256
    unless hashes.keys.to_set == expected_hashes.keys.to_set &&
           expected_hashes.all? { |name, sha| hashes[name].as_s == sha }
      raise ArgumentError.new("Full DiT upstream source digest inventory mismatch")
    end
    {"batches" => 1, "video_tokens" => 5280, "text_tokens" => 1280,
     "valid_text_tokens" => 23, "padded_text_tokens" => 1257,
     "valid_joint_tokens" => 5303, "total_tokens" => 6560,
     "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192,
     "double_blocks" => 54, "single_blocks" => 0, "query_chunk" => 256,
     "segments" => 1, "chunk_size_tokens" => 0}.each do |key, value|
      raise ArgumentError.new("Full DiT geometry mismatch: #{key}") unless doc["geometry"][key].as_i == value
    end
    g = doc["geometry"]
    unless g["rope_grid"].as_a.map(&.as_i) == [6, 22, 40] &&
           g["rope_axis_dims"].as_a.map(&.as_i) == [16, 56, 56] && g["rope_theta"].as_f == 256.0 &&
           doc["mask"].as_a.map(&.as_i) == MASK && doc["timestep"].as_f == 500.0
      raise ArgumentError.new("Full DiT RoPE, mask, or timestep mismatch")
    end
    p = doc["precision"]
    unless p["compute"].as_s == "F32" && p["global_values"].as_s == "F32->BF16->F32 on MPS" &&
           p["block_values_bf16"].as_s == "F32->BF16->F32 on MPS" &&
           p["block_values_q4"].as_s == "Q4_K 8 ranges decoded on MPS, rounded to BF16, represented F32" &&
           p["attention"].as_s == "PyTorch MPS SDPA F32 valid-prefix adapter" &&
           !p["native_bf16_compute"].as_bool && !p["cuda_magi_parity"].as_bool
      raise ArgumentError.new("Full DiT precision scope mismatch")
    end
    qwen = doc["qwen_source"]
    unless qwen["reference_sha256"].as_s == "eb64e5c31bc2178436b2b7dd411af45dea7fa43a8e3ff37a146bc411fac03ba1" &&
           qwen["sidecar_sha256"].as_s == "e7ebc9a784124afc20503435d84d42e0110d7c02054d3fc63fd586ec69ed4995" &&
           qwen["state_index"].as_i == 26 && qwen["crop"].as_a.map(&.as_i) == [108, 1388]
      raise ArgumentError.new("Full DiT saved Qwen producer boundary mismatch")
    end
    producer = doc["producer"]
    unless producer["device"].as_s == "mps" && producer["block_invocations"].as_a.map(&.as_i) == (0...54).to_a &&
           producer["terminal_activation_readbacks"].as_i == 1 &&
           producer["attention_adapter_calls"].as_i == 54 &&
           producer["query_chunks_per_block"].as_i == 21 &&
           producer["attention_query_chunk_calls"].as_i == 1134 &&
           producer["padding_poison_control_passed"].as_bool &&
           producer["captured_block_outputs"].as_a.map(&.as_i) == CAPTURE_BLOCKS &&
           producer["camera_conditioning"].as_s == "off" && producer["action_conditioning"].as_s == "off" &&
           !producer["full_generation"].as_bool
      raise ArgumentError.new("Full DiT producer execution/scope mismatch")
    end
    tensor_meta = doc["tensors"].as_h
    expected = shapes
    raise ArgumentError.new("Full DiT selected tensor inventory mismatch") unless tensor_meta.keys.to_set == expected.keys.to_set
    cursor = 0_i64
    expected.keys.sort.each do |name|
      meta = tensor_meta[name]
      count = expected[name].reduce(1_i64) { |a, b| a * b }
      unless meta["dtype"].as_s == "F32" && meta["shape"].as_a.map(&.as_i) == expected[name] &&
             meta["offset_bytes"].as_i64 == cursor && meta["nbytes"].as_i64 == count * 4
        raise ArgumentError.new("Full DiT tensor shape/layout mismatch: #{name}")
      end
      cursor += count * 4
    end
    unless cursor == SIDECAR_BYTES && doc["tensor_file"]["nbytes"].as_i64 == cursor &&
           doc["tensor_file"]["encoding"].as_s == "f32-le"
      raise ArgumentError.new("Full DiT sidecar extent/encoding mismatch")
    end
  end

  class Fixture
    getter document : JSON::Any
    getter path : String
    getter sha256 : String
    @sidecar : String
    @sidecar_info : File::Info
    @document_info : File::Info

    def initialize(@path : String, expected_sha : String, mode : String,
                   index : ML::Video::HyNomasQ4StackIndex::Index)
      @document_info = HyNomasQ4DiTIO.regular!(@path)
      raise ArgumentError.new("Full DiT fixture JSON is too large") unless 0 < @document_info.size <= 2 * 1024 * 1024
      ML::Video::HyNomasQ4Cache::Integrity.validate_sha256!(expected_sha, "fixture")
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("Full DiT fixture digest mismatch") unless @sha256 == expected_sha
      HyNomasQ4DiTIO.unchanged!(@path, @document_info)
      @document = JSON.parse(raw)
      HyNomasQ4DiTProbe.validate_document!(@document, mode, index)
      name = @document["tensor_file"]["name"].as_s
      unless name.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.f32le\z/)
        raise ArgumentError.new("Full DiT sidecar must be a safe sibling basename")
      end
      @sidecar = File.join(File.dirname(@path), name)
      @sidecar_info = HyNomasQ4DiTIO.regular!(@sidecar)
      unless @sidecar_info.size == SIDECAR_BYTES &&
             HyNomasQ4DiTIO.sha256_file(@sidecar) == @document["tensor_file"]["sha256"].as_s
        raise ArgumentError.new("Full DiT sidecar size/digest mismatch")
      end
      unchanged!
    end

    def unchanged! : Nil
      HyNomasQ4DiTIO.unchanged!(@path, @document_info)
      HyNomasQ4DiTIO.unchanged!(@sidecar, @sidecar_info)
    end

    def read_tensor(name : String, & : File ->) : Nil
      unchanged!
      File.open(@sidecar, "rb") do |io|
        io.seek(@document["tensors"][name]["offset_bytes"].as_i64)
        yield io
      end
      unchanged!
    end

    # Setup-only raw byte transfer. Expected DiT outputs cannot enter inference.
    def upload_input!(name : String, arena : ML::Video::Arena) : Nil
      raise ArgumentError.new("Only declared raw inputs can be uploaded") unless name.starts_with?("input.")
      bytes = @document["tensors"][name]["nbytes"].as_i64
      arena.alloc(name, (bytes // 4).to_i32)
      read_tensor(name) { |io| io.read_fully(Bytes.new(arena[name].contents.as(UInt8*), bytes.to_i32)) }
    end
  end

  def dit_config(compact_text : Bool = false, compact_text_tokens : Int32 = 23) : ML::Video::HyNomasConfig
    if compact_text
      CONFIG.compact_valid_text_prefix(23, compact_text_tokens)
    else
      raise ArgumentError.new("Compact storage requires compact text mode") unless compact_text_tokens == 23
      CONFIG
    end
  end

  def serial_bytes(compact_text : Bool = false, compact_text_tokens : Int32 = 23) : Int64
    c = dit_config(compact_text, compact_text_tokens)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(c).sum { |_, n| n.to_i64 * 4 }
    scratch += 53_i64 * c.joint_rows * c.dim * 4
    compact = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 5303, 16, 128, 256, 16)
    scratch + ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(c).to_i64 * 4 +
      # Q/K RMS keeps the source MPSGraph mean geometry so changing storage
      # does not change its reduction arithmetic. Other text scratch is M rows.
      compact.resident_bytes + 2_i64 * CONFIG.joint_rows * (c.dim + c.heads) * 4 +
      (compact_text ? ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(CONFIG, 23, compact_text_tokens) : 0_i64)
  end

  def boundary_bytes : Int64
    patch = (5280_i64 * 32 + 5280_i64 * 2048 + 2048_i64 * 32) * 4
    final = (2048_i64 + 2_i64 * 5280 * 2048 + 4096 + 2_i64 * 5280 * 32 + 2_i64 * 2048 * 2048) * 4
    patch + final
  end

  # Pure metadata admission for a pair of resident Q4 sessions. This must run
  # before constructing either model; actual allocations are checked again by
  # Session against the authenticated loader's independent admitted total.
  def q4_named_bytes(index : ML::Video::HyNomasQ4StackIndex::Index,
                     compact_text : Bool = false, compact_text_tokens : Int32 = 23) : Int64
    globals = HyNomasQ4DiTIO::GLOBAL_BINDINGS.sum do |binding|
      binding.shape.reduce(1_i64) { |a, b| a * b } * 2
    end
    inputs = shapes.select { |name, _| name.starts_with?("input.") }.values.sum do |shape|
      shape.reduce(1_i64) { |a, b| a * b } * 4
    end
    index.caches.sum(&.payload_bytes) + STAGE_BYTES + globals + inputs +
      1280_i64 * 3584 * 2 + boundary_bytes + serial_bytes(compact_text, compact_text_tokens) +
      ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(MASK, retain_padding: true).total +
      (compact_text ? 1280_i64 * 2048 * 4 : 0_i64)
  end

  class Session
    @@live = [] of Session
    getter named_bytes : Int64 = 0_i64
    getter cache_payload_bytes : Int64 = 0_i64
    getter poison_count : Int32 = 0
    getter compact_text : Bool
    getter dit_text_tokens : Int32
    @inputs = ML::Video::Arena.new
    @globals = ML::Video::Arena.new
    @stage = ML::Video::Arena.new
    @global_file : ML::Video::CheckpointFile? = nil
    @prepared : ML::Video::PreparedWeights? = nil
    @loader : (ML::Video::HyNomasQ4Stack | ML::Video::HyNomasBF16Stack)? = nil
    @patch : ML::Video::HyNomasPatchMPSPlan? = nil
    @refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
    @serial : ML::Video::HyNomasMPSSerialWorkspace? = nil
    @final : ML::Video::HyNomasFinalMPSPlan? = nil
    @convert : ML::Metal::ComputePipeline? = nil
    @poison : ML::Metal::ComputePipeline? = nil
    @poison_buffers = [] of ML::MetalBuffer
    @captures = {} of String => ML::MetalBuffer
    @first_hashes = {} of String => String
    @diagnostic = Array(Float32).new(5280 * 2048, 0_f32)
    @slab = Bytes.new(64 * 1024)
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@fixture : Fixture, index_path : String, index_sha : String,
                   index : ML::Video::HyNomasQ4StackIndex::Index, @mode : String,
                   budget : Int64, *, @compact_text : Bool = false,
                   compact_text_tokens : Int32 = 23)
      # All headers, payloads, independent BF16 controls, and aggregate sizes
      # are admitted before constructing the first MetalBuffer.
      @dit_text_tokens = HyNomasQ4DiTProbe.dit_config(@compact_text, compact_text_tokens).text_tokens
      file = ML::Video::BF16WeightCache.authenticate(index.globals_bf16_path, index.globals_bf16_sha256)
      @global_file = file
      begin
        bf16_bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |binding|
          ML::Video::WeightBinding.new(binding.target, binding.source, binding.shape, ML::GGUF::SafeTensorDType::BF16)
        end
        plan = ML::Video::WeightPlan.new([file], bf16_bindings,
          budget, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        unless plan.entries.size == 48 && plan.staging_bytes == 0 && plan.resident_dtype.bf16?
          raise ArgumentError.new("Full DiT globals require exact48 BF16 cache with no conversion stage")
        end
        input_bytes = HyNomasQ4DiTProbe.shapes.select { |name, _| name.starts_with?("input.") }.values.sum do |shape|
          shape.reduce(1_i64) { |a, b| a * b } * 4
        end
        reserve = plan.destination_bytes + input_bytes + 1280_i64 * 3584 * 2 +
                  HyNomasQ4DiTProbe.boundary_bytes + HyNomasQ4DiTProbe.serial_bytes(@compact_text, compact_text_tokens) +
                  (@compact_text ? 1280_i64 * 2048 * 4 : 0_i64) +
                  ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(MASK, retain_padding: true).total
        loader = if @mode == "q4"
                   ML::Video::HyNomasQ4Stack.new(index_path, index_sha, HyNomasQ4DiTIO::SOURCE_SHA,
                     CONFIG, budget, stage_bytes: STAGE_BYTES, reserve_bytes: reserve)
                 else
                   ML::Video::HyNomasBF16Stack.new(index_path, index_sha, HyNomasQ4DiTIO::SOURCE_SHA,
                     CONFIG, budget, stage_bytes: STAGE_BYTES, reserve_bytes: reserve)
                 end
        @loader = loader
        @named_bytes = loader.admitted_bytes
        @cache_payload_bytes = loader.cache_payload_bytes
        @@live << self
        @prepared = ML::Video::PreparedWeights.new(plan, @globals)
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG, block_index: 0).each do |binding|
          count = binding.shape.reduce(1_i64) { |a, b| a * b }
          @stage.alloc_weight(binding.target, count.to_i32, ML::Video::WeightDType::BF16)
        end
        loader.load!(@stage)
        HyNomasQ4DiTProbe.shapes.keys.select(&.starts_with?("input.")).each { |name| @fixture.upload_input!(name, @inputs) }
        @inputs.alloc_weight("qwen_bf16", 1280 * 3584, ML::Video::WeightDType::BF16)
        # Diagnostic backup is never an inference input. Only the excluded-tail
        # poison control uses it, outside performance samples, entirely on GPU.
        @inputs.alloc("diagnostic.text_tail_backup", 1280 * 2048) if @compact_text
        @convert = ML::Metal::ComputePipeline.new("video_weight_encode_bf16", ML::Video::PreparedWeights::SOURCE)
        @patch = patch = ML::Video::HyNomasPatchMPSPlan.new(@inputs["input.latent"], 5280, @globals)
        @refiner = refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(@inputs["qwen_bf16"], MASK, 500_f32, @globals,
          ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(MASK, retain_padding: true).total, retain_padding: true)
        @serial = serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG,
          Array(ML::Video::Arena).new(54) { @stage }, patch.output, refiner.text_output,
          refiner.vec, refiner.vec_txt, @inputs["input.rope_cos"], @inputs["input.rope_sin"],
          attention_query_chunk: 256, attention_heads_per_group: 16, valid_text_tokens: 23,
          allow_shared_modulation: true, full_dit: true, weight_loader: loader,
          compact_valid_text: @compact_text,
          compact_text_tokens: @compact_text ? compact_text_tokens : nil)
        last = serial.blocks.last
        @final = final = ML::Video::HyNomasFinalMPSPlan.new(last.image_output, refiner.vec, 5280, @globals)
        @captures["img_in"] = patch.output
        @captures["conditioning.projected_text"] = refiner.text_output
        @captures["conditioning.vec"] = refiner.vec
        CAPTURE_BLOCKS.each do |i|
          @captures["block#{i}.img.output"] = serial.blocks[i].image_output
          @captures["block#{i}.txt.output"] = serial.blocks[i].text_output
        end
        final.checkpoints.each { |name, buffer| @captures[name] = buffer }
        unless @captures.keys.sort == HyNomasQ4DiTProbe.shapes.keys.reject(&.starts_with?("input.")).sort
          raise "Full DiT native capture inventory mismatch"
        end
        @captures.each do |name, buffer|
          expected_bytes = if @compact_text && name.starts_with?("block") && name.ends_with?(".txt.output")
                             @dit_text_tokens.to_i64 * 2048 * 4
                           else
                             @fixture.document["tensors"][name]["nbytes"].as_i64
                           end
          raise "Full DiT capture extent mismatch: #{name}" unless buffer.size == expected_bytes
        end
        actual_bytes = @cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes + @inputs.resident_bytes +
                       patch.resident_bytes + refiner.resident_bytes + serial.resident_bytes + final.resident_bytes
        raise "Full DiT allocation differs from admitted bytes: #{actual_bytes} vs #{@named_bytes}" unless actual_bytes == @named_bytes
        # Poison producer/consumer scratch between complete requests, including
        # distinct output banks, but preserve immutable inputs and all weights.
        @poison_buffers << patch.output
        refiner.checkpoints.each do |name, buffer|
          next if name == "timestep"
          @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer))
        end
        serial.blocks.each do |block|
          scratch, _ = block.mps_workspace_buffers
          scratch.each_with_index do |buffer, slot|
            next if ML::Video::HyNomasDualStreamBlock.scratch_plan(serial.block_config)[slot][0] == "hy_nomas.zero_head_bias"
            @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer))
          end
        end
        if buffer = serial.compact_text_input
          @poison_buffers << buffer
        end
        serial.hy_dit_probe_rms_scratch.each do |buffer|
          @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer))
        end
        @poison_buffers << @inputs["diagnostic.text_tail_backup"] if @compact_text
        final.checkpoints.each_value { |buffer| @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer)) }
        @poison_count = @poison_buffers.size
        @poison = ML::Metal::ComputePipeline.new("hy_nomas_dit_poison_f32", <<-METAL)
          #include <metal_stdlib>
          using namespace metal;
          kernel void hy_nomas_dit_poison_f32(device float *p [[buffer(0)]], constant uint &n [[buffer(1)]], constant uint &offset [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < n) p[offset + i] = as_type<float>(0x7fc00000u);
          }
          METAL
      rescue ex
        release
        raise ex
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain, *, poison : Bool,
               poison_text_suffix : Bool = false) : Nil
      raise ArgumentError.new("Full DiT session has been released") if @released
      raise ArgumentError.new("Excluded-tail poison requires compact DiT") if poison_text_suffix && !@compact_text
      if previous = @last_chain
        raise ArgumentError.new("Full DiT previous command chain is pending") unless previous.completed? && !previous.same?(chain)
      end
      # Read-only complete block/loader preflight before any producer appends.
      @patch.not_nil!.validate_for!(chain)
      @refiner.not_nil!.validate_for!(chain)
      @serial.not_nil!.validate_for!(chain)
      @final.not_nil!.validate_for!(chain)
      @fixture.unchanged!
      @last_chain = chain
      @prepared.not_nil!.encode(chain.encoder)
      if poison
        @poison_buffers.each do |buffer|
          chain.encoder.set_pipeline(@poison.not_nil!).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
            .set_value((buffer.size // 4).to_u32, 1).set_value(0_u32, 2).dispatch_1d((buffer.size // 4).to_i32)
        end
      end
      chain.encoder.set_pipeline(@convert.not_nil!).set_buffer(@inputs["input.qwen_hidden"], 0)
        .set_buffer(@inputs["qwen_bf16"], 1, ML::Metal::BufferAccess::Write)
        .set_value((1280 * 3584).to_u32, 2).dispatch_1d(1280 * 3584)
      @patch.not_nil!.encode(chain)
      @refiner.not_nil!.encode(chain)
      if poison_text_suffix
        ML::Video::MetalOps.copy(chain.encoder, @refiner.not_nil!.text_output,
          @inputs["diagnostic.text_tail_backup"], 1280 * 2048)
        chain.encoder.set_pipeline(@poison.not_nil!).set_buffer(@refiner.not_nil!.text_output, 0, ML::Metal::BufferAccess::Write)
          .set_value((1257 * 2048).to_u32, 1).set_value((23 * 2048).to_u32, 2).dispatch_1d(1257 * 2048)
      end
      @serial.not_nil!.encode(chain)
      if poison_text_suffix
        ML::Video::MetalOps.copy(chain.encoder, @inputs["diagnostic.text_tail_backup"],
          @refiner.not_nil!.text_output, 1280 * 2048)
      end
      @final.not_nil!.encode(chain)
    end

    def compare(request : Int32) : Array(Comparison)
      chain = @last_chain || raise ArgumentError.new("Full DiT has not encoded")
      raise ArgumentError.new("Full DiT diagnostics require terminal wait") unless chain.completed? && chain.wait_count == 1
      @captures.keys.sort.map do |name|
        buffer = @captures[name]
        # Aligned storage may retain excluded rows. Only the valid prefix is a
        # semantic output; poisoned suffix rows are deliberately not read.
        count = if @compact_text && name.starts_with?("block") && name.ends_with?(".txt.output")
                  23 * 2048
                else
                  (buffer.size // 4).to_i32
                end
        actual = Slice.new(@diagnostic.to_unsafe, count)
        buffer.read_to_slice(actual)
        hash = Digest::SHA256.hexdigest(Bytes.new(actual.to_unsafe.as(UInt8*), count * 4))
        repeat_match = request == 0 ? nil : @first_hashes[name] == hash
        @first_hashes[name] = hash if request == 0
        failures, differences, cursor = 0, 0, 0
        max_abs = 0_f64
        @fixture.read_tensor(name) do |io|
          while cursor < count
            take = Math.min(@slab.size // 4, count - cursor)
            slab = @slab[0, take * 4]
            io.read_fully(slab)
            take.times do |i|
              expected = IO::ByteFormat::LittleEndian.decode(Float32, slab[i * 4, 4])
              value = actual[cursor + i]
              differences += 1 if value.unsafe_as(UInt32) != expected.unsafe_as(UInt32)
              if value.finite? && expected.finite?
                error = (value.to_f64 - expected.to_f64).abs
                max_abs = Math.max(max_abs, error)
                failures += 1 if error > ATOL + RTOL * expected.to_f64.abs
              else
                failures += 1
                max_abs = Float64::INFINITY
              end
            end
            cursor += take
          end
        end
        {name: name, elements: count, failures: failures, raw_bit_differences: differences,
         max_abs: max_abs.finite? ? max_abs : nil, finite: max_abs.finite?,
         native_sha256: hash, first_request_sha256_match: repeat_match}
      end
    end

    def release : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Complete the full DiT command chain before release") unless previous.completed?
      end
      @final.try(&.release)
      @serial.try(&.release)
      @refiner.try(&.release)
      @patch.try(&.release)
      @loader.try(&.release)
      @prepared.try(&.release)
      @stage.release
      @globals.release
      @inputs.release
      @global_file.try(&.close)
      @captures.clear
      @poison_buffers.clear
      @released = true
      @@live.reject!(&.same?(self))
    end
  end
end
