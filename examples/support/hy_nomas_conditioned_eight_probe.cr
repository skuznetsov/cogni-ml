require "./hy_nomas_four_block_probe"
require "../../src/ml/video/hy_nomas_text_refiner"

# Probe-only composition of resident Qwen conditioning, the real text refiner,
# and eight actual DiT blocks. Expected conditioning is never an inference input.
module HyNomasConditionedEightProbe
  extend self
  alias Four = HyNomasFourBlockProbe
  alias Serial = HyNomasPretrainedSerialProbe
  CONFIG            = Four::CONFIG
  SCHEMA            = "hy-nomas-conditioned-eight-reference-v1"
  TEXT_SHA          = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
  TEXT_MANIFEST_SHA = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
  VALID_TEXT        =              23
  VALID_JOINT       =            5303
  SIDECAR_BYTES     = 489_070_592_i64
  INPUT_BYTES       = (5280_i64 * 2048 + 2_i64 * 5280 * 128) * 4
  MAX_SESSION_BYTES = 8_i64 * 1024 * 1024 * 1024
  CACHE_SHAS        = Four::CACHE_SHAS + [
    "3d6a84e8884ed33c4c90203f36d5c580e620e58c0e62c152c89ea5a9879b4fde",
    "727201b5361c4589d2f38293fc320a3efb64ca3f2ceed34426cef54b149d0c69",
    "dd81d9d3aabb51b62ab0658eb4ae635d37d8d532a576815a27e2b433c1aa18d2",
    "df134b944627c037caf2931ea3e4e6c37642221ea964aae15d0c37073865ffc4",
  ]
  SOURCE_PINS = Four::SOURCE_PINS + [
    {file_sha256: "08a7502785d0cac541f83a7931494eceaae8c7cef5ff7212b7710a92a5dcfb5f", manifest_sha256: "e52d4e096f5cc972db3cca709398283898a82cb11889852002e180340b80f0ac"},
    {file_sha256: "140ffda85831042315ff0c21b9e6bc2df7ad523508aed0cd065c43e07cca5e3f", manifest_sha256: "036980294835bf4fa7e803d834a25a2266c5dd4feb03398b8c1251c7dd404ad8"},
    {file_sha256: "90c135deb5d0f0031ed5e69e6032667bdf4a91edbb87396203ea6880d2d73e22", manifest_sha256: "9cdd4a9274eea986935d0c54b34e8412de2ba5239dee471d904142ec89dada77"},
    {file_sha256: "658bfc21ddd09a5ca03130e08fcdcece16fc4ce75bfcab5ec2c969caef5218d5", manifest_sha256: "92d1216d02de9fa5eed098fe1ac73d21c767ed49aed60ea3fb395006011c86a2"},
  ]
  UPSTREAM_SOURCE_SHA256 = {
    "hymm/models/dit/nomas_cpt.py"                       => "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    "hymm/models/dit/modules/activation_layers.py"       => "1e509b5e5f9527d199d6f0206f452e56fdb1bd25ce3ccc06fb47f3a16060ca39",
    "hymm/models/dit/modules/norm_layers.py"             => "2bcd5b29c951051ee1afaeac08258a3ef1dea3205b821f236594719331127044",
    "hymm/models/dit/modules/modulate_layers.py"         => "abfb87acc9a972b40d0f36cfa832f0816d978fc13f93236aa1dfc902aac5caee",
    "hymm/models/dit/modules/mlp_layers.py"              => "b77d862d17cd2c7db6237931b80f33b6447b7ba4db5918d3fbdfd17a6fb47e07",
    "hymm/models/dit/modules/posemb_layers.py"           => "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/dit/modules/attention/__init__.py"      => "2a59284ac06057ac622228d280a32fc9e907a333e8ef0efcd526928626858450",
    "hymm/models/dit/modules/attention/magi_attn.py"     => "3cdd9234db6ddd006a04af53b878a6876242ed4fa352bbcfe43038519492eb47",
    "hymm/utils/helpers.py"                              => "51b7a08377843b5eecc59e6fd41f10317d2de096d30fc3154491aacd27fefdfb",
    "hymm/models/text_encoder/token_refiner.py"          => "7c67d81b81d9b183c3261eb4bf87bb04e0bda3142ce096afc2e64aa722433eab",
    "hymm/models/dit/modules/embed_layers.py"            => "5b055e383bbf532ea5aa49ee506031d96bea3e3d5b8da3a104d4bd875c2d0af3",
    "hymm/models/dit/modules/int8_stored_bf16_linear.py" => "2bfd28f483c0da76df5ece7b23d0d3f62ef38725317b13e55d41ce997812122a",
  }
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    raw_bit_differences: Int32, max_abs: Float64?, finite: Bool,
    native_sha256: String, first_request_sha256_match: Bool?)

  def shapes : Hash(String, Array(Int32))
    result = {"input.img" => [1, 5280, 2048],
              "input.rope_cos" => [5280, 128], "input.rope_sin" => [5280, 128],
              "conditioning.projected_text" => [1, 1280, 2048],
              "conditioning.vec" => [1, 2048]}
    8.times do |index|
      result["block#{index}.img.output"] = [1, 5280, 2048]
      result["block#{index}.txt.output"] = [1, 1280, 2048]
    end
    result
  end

  def validate_document!(doc : JSON::Any, qwen_sha : String) : Nil
    unless doc["schema"].as_s == SCHEMA && doc["schema_version"].as_i == 1 &&
           doc["upstream"]["revision"].as_s == Serial::PINNED_UPSTREAM_REVISION
      raise ArgumentError.new("conditioned eight-block schema/upstream mismatch")
    end
    source_hashes = doc["upstream"]["source_sha256"].as_h
    unless source_hashes.keys.to_set == UPSTREAM_SOURCE_SHA256.keys.to_set &&
           UPSTREAM_SOURCE_SHA256.all? { |name, sha| source_hashes[name].as_s == sha }
      raise ArgumentError.new("conditioned eight-block upstream source digest inventory mismatch")
    end
    {"batches" => 1, "video_tokens" => 5280, "text_tokens" => 1280,
     "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192,
     "query_chunk" => 256, "attention_segments" => 1, "chunk_size_tokens" => 0}.each do |name, value|
      raise ArgumentError.new("conditioned eight-block geometry mismatch: #{name}") unless doc["geometry"][name].as_i == value
    end
    {"valid_text_tokens" => VALID_TEXT, "padded_text_tokens" => 1257,
     "valid_joint_tokens" => VALID_JOINT, "refiner_input_dim" => 3584,
     "refiner_depth" => 2}.each do |name, value|
      raise ArgumentError.new("conditioned padding geometry mismatch: #{name}") unless doc["geometry"][name].as_i == value
    end
    g = doc["geometry"]
    unless g["rope_grid"].as_a.map(&.as_i) == [6, 22, 40] &&
           g["rope_axis_dims"].as_a.map(&.as_i) == [16, 56, 56] && g["rope_theta"].as_f == 256.0
      raise ArgumentError.new("conditioned eight-block RoPE mismatch")
    end
    unless doc["tolerance"]["atol"].as_f == Serial::ATOL && doc["tolerance"]["rtol"].as_f == Serial::RTOL &&
           doc["precision"]["compute"].as_s == "F32" &&
           doc["precision"]["parameter_storage"].as_s == "F32" &&
           doc["precision"]["attention"].as_s == "PyTorch MPS SDPA F32" &&
           doc["precision"]["parameter_values"].as_s == "BF16-rounded values" &&
           doc["precision"]["source_weight_dtype"].as_s == "F32" &&
           doc["precision"]["weight_rounding"].as_s == "source F32 values converted F32->BF16->F32 on MPS" &&
           !doc["precision"]["native_bf16_parameter_storage"].as_bool &&
           !doc["precision"]["native_bf16_compute"].as_bool &&
           !doc["precision"]["cuda_bf16_parity"].as_bool
      raise ArgumentError.new("conditioned eight-block precision/unchanged tolerance mismatch")
    end
    receipt = doc["producer"]
    {"conditioning_invocations" => 1, "refiner_rows" => 1280, "refiner_attention_calls" => 2,
     "attention_adapter_calls" => 8, "attention_query_chunks_per_block" => 21,
     "attention_query_chunk_calls" => 168, "terminal_tensor_count" => 21,
     "padding_poison_control_calls" => 1, "padding_poison_control_query_chunks" => 21}.each do |name, value|
      raise ArgumentError.new("conditioned producer receipt mismatch: #{name}") unless receipt[name].as_i == value
    end
    unless receipt["device"].as_s == "mps" &&
           receipt["activation_readback"].as_s == "one packed final F32 tensor sidecar readback" &&
           receipt["attention_adapter"].as_s == "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256" &&
           receipt["block_invocations"].as_a.map(&.as_s) == (0...8).map { |i| "block#{i}" } &&
           receipt["captures_per_block"].as_a.map(&.as_s) == Four::CAPTURE_NAMES &&
           receipt["padding_poison_control_passed"].as_bool &&
           receipt["max_attention_query_chunk"].as_i == 256 &&
           receipt["block_handoff"]["img"].as_s == "each block output passed directly as next block input" &&
           receipt["block_handoff"]["txt"].as_s == "each block output passed directly as next block input" &&
           receipt["block_handoff"]["same_mps_tensor_objects"].as_bool &&
           receipt["terminal_sidecar_bytes"].as_i64 == SIDECAR_BYTES
      raise ArgumentError.new("conditioned producer device/readback/handoff/control receipt mismatch")
    end
    q = doc["qwen_source"]
    unless qwen_sha.matches?(/\A[0-9a-f]{64}\z/) && q["reference_sha256"].as_s == qwen_sha &&
           q["sidecar_sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/) && q["state_index"].as_i == 26 &&
           q["crop"].as_a.map(&.as_i) == [108, 1388] && doc["timestep"].as_f == 500.0 &&
           doc["mask"].as_a.map(&.as_i) == Array(Int32).new(1280) { |i| i < VALID_TEXT ? 1 : 0 }
      raise ArgumentError.new("conditioned eight-block Qwen crop/mask/timestep mismatch")
    end
    p = doc["padding_policy"]
    unless p["kind"].as_s == "unpad_zero_tokens" && p["valid_joint_prefix"].as_i == VALID_JOINT &&
           p["valid_text_rows"].as_i == VALID_TEXT && p["padded_text_rows"].as_i == 1257 &&
           p["refiner_rows"].as_i == 1280 && p["padded_query_outputs"].as_s == "attention result explicitly zero-filled"
      raise ArgumentError.new("conditioned eight-block padding policy mismatch")
    end
    input_generation = doc["input_generation"]
    rope_generation = input_generation["rope"]
    unless input_generation["kind"].as_s == "deterministic synthetic MPS F32 video input plus pinned upstream ND RoPE" &&
           input_generation["video_formula"].as_s == "x[i] = 0.1*sin(float32(i)*0.017+phase) + 0.05*cos(float32(i)*0.0031+phase*0.7)" &&
           input_generation["video_phase"].as_f == 0.11 &&
           rope_generation["implementation"].as_s == "pinned upstream posemb_layers.get_nd_rotary_pos_embed" &&
           rope_generation["grid"].as_a.map(&.as_i) == [6, 22, 40] &&
           rope_generation["grid_axis_order"].as_a.map(&.as_s) == ["temporal", "height", "width"] &&
           rope_generation["axis_dims"].as_a.map(&.as_i) == [16, 56, 56] &&
           rope_generation["theta"].as_f == 256.0 && rope_generation["use_real"].as_bool &&
           rope_generation["device"].as_s == "mps"
      raise ArgumentError.new("conditioned eight-block input-generation recipe mismatch")
    end
    unless doc["text_source"]["sha256"].as_s == TEXT_SHA && doc["text_source"]["manifest_sha256"].as_s == TEXT_MANIFEST_SHA &&
           doc["text_source"]["nbytes"].as_i64 == 583_316_840_i64 && doc["text_source"]["tensor_count"].as_i == 42
      raise ArgumentError.new("conditioned eight-block refiner source mismatch")
    end
    source = doc["source"]
    block_names = (0...8).map { |i| "block#{i}" }
    unless source["block_order"].as_a.map(&.as_s) == block_names &&
           source["blocks"].as_h.keys.to_set == block_names.to_set
      raise ArgumentError.new("conditioned eight-block source order/inventory mismatch")
    end
    SOURCE_PINS.each_with_index do |pin, index|
      block = source["blocks"]["block#{index}"]
      unless block["block_index"].as_i == index && block["file_sha256"].as_s == pin[:file_sha256] &&
             block["manifest_sha256"].as_s == pin[:manifest_sha256] &&
             block["tensor_prefix"].as_s == "double_blocks.#{index}." && block["tensor_count"].as_i == 32
        raise ArgumentError.new("conditioned eight-block source pin mismatch: #{index}")
      end
    end
    expected = shapes
    tensors = doc["tensors"].as_h
    raise ArgumentError.new("conditioned eight-block requires twenty-one selected tensors") unless tensors.keys.to_set == expected.keys.to_set
    cursor = 0_i64
    expected.keys.sort.each do |name|
      meta = tensors[name]
      bytes = expected[name].reduce(1_i64) { |a, b| a * b } * 4
      unless meta["dtype"].as_s == "F32" && meta["shape"].as_a.map(&.as_i) == expected[name] &&
             meta["offset_bytes"].as_i64 == cursor && meta["nbytes"].as_i64 == bytes
        raise ArgumentError.new("conditioned eight-block tensor layout mismatch: #{name}")
      end
      cursor += bytes
    end
    unless cursor == SIDECAR_BYTES && doc["tensor_file"]["nbytes"].as_i64 == cursor &&
           doc["tensor_file"]["encoding"].as_s == "f32-le" && doc["tensor_file"]["sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("conditioned eight-block sidecar extent/encoding mismatch")
    end
  end

  class Fixture
    getter document : JSON::Any
    getter sha256 : String
    getter path : String
    getter text_weights : String
    @sidecar : String
    @info : File::Info

    def initialize(@path : String, sha : String, qwen_sha : String)
      unless File.file?(@path) && !File.symlink?(@path) && 0 < File.size(@path) <= 2 * 1024 * 1024 && sha.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("conditioned reference requires a bounded regular JSON and trusted SHA256")
      end
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("conditioned reference digest mismatch") unless @sha256 == sha
      @document = JSON.parse(raw)
      HyNomasConditionedEightProbe.validate_document!(@document, qwen_sha)
      @text_weights = @document["text_source"]["path"].as_s
      unless File.file?(@text_weights) && !File.symlink?(@text_weights) &&
             File.size(@text_weights) == @document["text_source"]["nbytes"].as_i64 && Serial.sha256_file(@text_weights) == TEXT_SHA
        raise ArgumentError.new("conditioned text weight digest/type/extent mismatch")
      end
      name = @document["tensor_file"]["name"].as_s
      raise ArgumentError.new("conditioned sidecar must be a sibling basename") unless name.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.f32le\z/)
      @sidecar = File.join(File.dirname(File.realpath(@path)), name)
      unless File.file?(@sidecar) && !File.symlink?(@sidecar) && File.size(@sidecar) == SIDECAR_BYTES
        raise ArgumentError.new("conditioned sidecar file extent/type mismatch")
      end
      @info = File.info(@sidecar)
      raise ArgumentError.new("conditioned sidecar digest mismatch") unless Serial.sha256_file(@sidecar) == @document["tensor_file"]["sha256"].as_s
      unchanged!
    end

    def mask : Array(Int32)
      @document["mask"].as_a.map(&.as_i)
    end

    def timestep : Float32
      @document["timestep"].as_f.to_f32
    end

    def unchanged! : Nil
      info = File.info(@sidecar, follow_symlinks: false)
      unless info.file? && @info.same_file?(info) && info.size == @info.size && info.modification_time == @info.modification_time
        raise ArgumentError.new("conditioned sidecar changed during use")
      end
    end

    def read_tensor(name : String, & : File ->) : Nil
      unchanged!
      File.open(@sidecar) do |io|
        io.seek(@document["tensors"][name]["offset_bytes"].as_i64)
        yield io
      end
      unchanged!
    end

    def values(name : String) : Array(Float32)
      raise ArgumentError.new("expected conditioning/captures cannot be uploaded as native inputs") unless name.starts_with?("input.")
      size = @document["tensors"][name]["nbytes"].as_i64.to_i32
      bytes = Bytes.new(size)
      read_tensor(name) { |io| io.read_fully(bytes) }
      values = Array(Float32).new(size // 4) { |i| IO::ByteFormat::LittleEndian.decode(Float32, bytes[i * 4, 4]) }
      raise ArgumentError.new("nonfinite conditioned input: #{name}") unless values.all?(&.finite?)
      values
    end
  end

  def validate_sources!(sources : Array(Serial::BF16Source)) : Nil
    raise ArgumentError.new("conditioned eight-block replay requires eight ordered BF16 sources") unless sources.size == 8
    sources.each_with_index do |source, index|
      unless source[:sha256] == CACHE_SHAS[index] && source[:sha256].matches?(/\A[0-9a-f]{64}\z/) &&
             File.file?(source[:path]) && !File.symlink?(source[:path])
        raise ArgumentError.new("conditioned cache is not the qualified block #{index}")
      end
      index.times do |previous|
        if File.info(source[:path]).same_file?(File.info(sources[previous][:path]))
          raise ArgumentError.new("conditioned eight-block BF16 source files alias")
        end
      end
    end
  end

  def validate_bf16_plan!(plan : ML::Video::WeightPlan, index : Int32) : Nil
    unless 0 <= index < 8 && plan.entries.size == 32 &&
           plan.entries.all? { |entry| entry.binding.source.starts_with?("double_blocks.#{index}.") && entry.info.dtype.bf16? } &&
           plan.resident_dtype.bf16? && plan.destination_bytes == 302_113_792_i64 &&
           !plan.allow_f32_to_bf16 && plan.staging_bytes == 0
      raise ArgumentError.new("conditioned eight-block BF16 plan requires its full exact-index inventory and zero conversion staging")
    end
  end

  def named_bytes(weight_bytes : Int64) : NamedTuple(scratch: Int64, workspace: Int64, total: Int64)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG).sum { |_, n| n.to_i64 * 4 }
    scratch += 7_i64 * CONFIG.joint_rows * CONFIG.dim * 4
    compact = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
      1, 6560, 16, 128, 256, 16)
    workspace = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(CONFIG).to_i64 * 4 +
                compact.resident_bytes + 2_i64 * CONFIG.joint_rows * (CONFIG.dim + CONFIG.heads) * 4
    {scratch: scratch, workspace: workspace,
     total: weight_bytes + INPUT_BYTES + scratch + workspace}
  end

  # Header/digest-only admission, before the live Qwen producer allocates Metal.
  def admit(fixture : Fixture, sources : Array(Serial::BF16Source), budget : Int64, mode : String) : Int64
    raise ArgumentError.new("unknown conditioned eight-block control") unless {"positive", "padding-as-valid", "block4-as-block3", "block7-as-block6"}.includes?(mode)
    validate_sources!(sources)
    files = [] of ML::Video::CheckpointFile
    begin
      weights_bytes = 0_i64
      sources.each_with_index do |source, index|
        raise ArgumentError.new("conditioned cache is not the qualified block #{index}") unless source[:sha256] == CACHE_SHAS[index]
        file = ML::Video::BF16WeightCache.authenticate(source[:path], source[:sha256])
        files << file
        plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, budget,
          resident_dtype: ML::Video::WeightDType::BF16, block_index: index.to_i32, require_all: true)
        validate_bf16_plan!(plan, index.to_i32)
        weights_bytes += plan.peak_bytes
      end
      text = ML::Video::CheckpointFile.new(fixture.text_weights)
      files << text
      text_plan = ML::Video::WeightPlan.new([text], ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
        budget, resident_dtype: ML::Video::WeightDType::BF16, require_all: true, allow_f32_to_bf16: true)
      attention_tokens = mode == "padding-as-valid" ? 6560 : VALID_JOINT
      sizes = named_bytes(weights_bytes)
      old_attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 6560, 16, 128, 256, 16).resident_bytes
      attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, attention_tokens, 16, 128, 256, 16).resident_bytes
      total = sizes[:total] - old_attention + attention + text_plan.peak_bytes +
              ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(fixture.mask, retain_padding: true).total
      unless 0 < total <= budget && total <= MAX_SESSION_BYTES
        raise ArgumentError.new("conditioned eight-block exceeds aggregate named-byte budget or 8 GiB session limit")
      end
      total
    ensure
      files.each(&.close)
    end
  end

  class Session
    @@live = [] of Session
    getter named_bytes : Int64
    getter poison_count : Int32 = 0
    @inputs : ML::Video::Arena? = nil
    @text_weights : ML::Video::Arena? = nil
    @refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
    @serial : ML::Video::HyNomasMPSSerialWorkspace? = nil
    @weights = [] of ML::Video::Arena
    @prepared = [] of ML::Video::PreparedWeights
    @files = [] of ML::Video::CheckpointFile
    @captures = {} of String => ML::MetalBuffer
    @poison_buffers = [] of ML::MetalBuffer
    @poison : ML::Metal::ComputePipeline? = nil
    @lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @released = false
    @first_hashes = {} of String => String
    @diagnostic = Array(Float32).new(5280 * 2048, 0_f32)
    @slab = Bytes.new(64 * 1024)

    def initialize(@fixture : Fixture, source : ML::MetalBuffer, mask : Array(Int32), sources : Array(Serial::BF16Source), budget : Int64, @mode : String)
      raise ArgumentError.new("conditioned mask differs from authenticated prompt") unless mask == @fixture.mask
      @named_bytes = HyNomasConditionedEightProbe.admit(@fixture, sources, budget, @mode)
      begin
        @@live << self
        sources.each_with_index do |item, index|
          file = ML::Video::BF16WeightCache.authenticate(item[:path], item[:sha256])
          @files << file
          plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, budget,
            resident_dtype: ML::Video::WeightDType::BF16, block_index: index.to_i32, require_all: true)
          HyNomasConditionedEightProbe.validate_bf16_plan!(plan, index.to_i32)
          arena = ML::Video::Arena.new
          @weights << arena
          @prepared << ML::Video::PreparedWeights.new(plan, arena)
        end
        text_file = ML::Video::CheckpointFile.new(@fixture.text_weights)
        @files << text_file
        text_plan = ML::Video::WeightPlan.new([text_file], ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
          budget, resident_dtype: ML::Video::WeightDType::BF16, require_all: true, allow_f32_to_bf16: true)
        text_weights = ML::Video::Arena.new
        @text_weights = text_weights
        @prepared << ML::Video::PreparedWeights.new(text_plan, text_weights)
        refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(source, mask, @fixture.timestep, text_weights,
          ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total, retain_padding: true)
        @refiner = refiner
        inputs = ML::Video::Arena.new
        @inputs = inputs
        {"input.img", "input.rope_cos", "input.rope_sin"}.each { |name| inputs.upload(name, @fixture.values(name)) }
        used = @weights.dup
        used[4] = @weights[3] if @mode == "block4-as-block3"
        used[7] = @weights[6] if @mode == "block7-as-block6"
        serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG, used, inputs["input.img"], refiner.text_output,
          refiner.vec, refiner.vec_txt, inputs["input.rope_cos"], inputs["input.rope_sin"],
          attention_query_chunk: 256, attention_heads_per_group: 16,
          valid_text_tokens: @mode == "padding-as-valid" ? 1280 : VALID_TEXT, allow_shared_modulation: true)
        @serial = serial
        @captures["conditioning.projected_text"] = refiner.text_output
        @captures["conditioning.vec"] = refiner.vec
        refiner.checkpoints.each do |name, buffer|
          next if name == "timestep"
          @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer))
        end
        serial.blocks.each_with_index do |block, index|
          Four::CAPTURE_NAMES.each { |name| @captures["block#{index}.#{name}"] = block.checkpoints[name] }
          scratch, _ = block.mps_workspace_buffers
          plan = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG)
          scratch.each_with_index do |buffer, slot|
            next if plan[slot][0] == "hy_nomas.zero_head_bias"
            @poison_buffers << buffer unless @poison_buffers.any?(&.same?(buffer))
          end
        end
        expected = HyNomasConditionedEightProbe.shapes.keys.reject(&.starts_with?("input.")).sort
        raise "conditioned selected capture inventory mismatch" unless @captures.keys.sort == expected
        @captures.each do |name, buffer|
          raise "conditioned capture extent mismatch: #{name}" unless buffer.size == @fixture.document["tensors"][name]["nbytes"].as_i64
        end
        total = @weights.sum(&.resident_bytes) + text_plan.staging_bytes + text_weights.resident_bytes +
                inputs.resident_bytes + refiner.resident_bytes + serial.resident_bytes
        raise "conditioned aggregate named-byte estimate differs from allocation" unless total == @named_bytes
        @poison = ML::Metal::ComputePipeline.new("hy_conditioned_poison", <<-METAL)
          #include <metal_stdlib>
          using namespace metal;
          kernel void hy_conditioned_poison(device float *p [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i < n) p[i] = as_type<float>(0x7fc00000u);
          }
          METAL
        @poison_count = @poison_buffers.size.to_i32
      rescue ex
        release
        raise ex
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain, *, poison : Bool) : Nil
      raise ArgumentError.new("conditioned session has been released") if @released
      @lease.begin_encode!(chain)
      @prepared.each { |p| p.encode(chain.encoder) }
      if poison
        @poison_buffers.each do |buffer|
          chain.encoder.set_pipeline(@poison.not_nil!).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
            .set_value((buffer.size // 4).to_u32, 1).dispatch_1d((buffer.size // 4).to_i32)
        end
      end
      @refiner.not_nil!.encode(chain)
      @serial.not_nil!.encode(chain)
    end

    def compare(request : Int32) : Array(Comparison)
      @lease.ensure_terminal!
      @captures.keys.sort.map do |name|
        buffer = @captures[name]
        count = (buffer.size // 4).to_i32
        actual = Slice.new(@diagnostic.to_unsafe, count)
        buffer.read_to_slice(actual)
        hash = Digest::SHA256.hexdigest(Bytes.new(actual.to_unsafe.as(UInt8*), count * 4))
        repeat_match = request == 0 ? nil : @first_hashes[name] == hash
        @first_hashes[name] = hash if request == 0
        failures = 0
        differences = 0
        max_abs = 0_f64
        cursor = 0
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
                failures += 1 if error > Serial::ATOL + Serial::RTOL * expected.to_f64.abs
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
      @lease.ensure_terminal!
      @serial.try(&.release)
      @refiner.try(&.release)
      @prepared.reverse_each(&.release)
      @inputs.try(&.release)
      @text_weights.try(&.release)
      @weights.reverse_each(&.release)
      @files.each(&.close)
      @captures.clear
      @poison_buffers.clear
      @released = true
      @@live.reject!(&.same?(self))
    end
  end

  def negative_detected?(mode : String, rows : Array(Comparison)) : Bool
    names = shapes.keys.reject(&.starts_with?("input.")).sort
    return false unless rows.map(&.[:name]).sort == names && rows.all?(&.[:finite])
    return false unless rows.select { |row| row[:name].starts_with?("conditioning.") }.all? { |row| row[:failures] == 0 }
    fault = case mode
            when "padding-as-valid" then 0
            when "block4-as-block3" then 4
            when "block7-as-block6" then 7
            else                         return false
            end
    (0...fault).all? do |index|
      rows.select { |row| row[:name].starts_with?("block#{index}.") }.all? { |row| row[:failures] == 0 }
    end && rows.select { |row| row[:name].starts_with?("block#{fault}.") }.all? { |row| row[:failures] > 0 }
  end
end
