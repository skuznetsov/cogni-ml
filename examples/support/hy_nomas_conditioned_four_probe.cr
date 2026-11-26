require "./hy_nomas_four_block_probe"
require "../../src/ml/video/hy_nomas_text_refiner"

# Probe-only composition of resident Qwen conditioning, the real text refiner,
# and four actual DiT blocks. Expected conditioning is never an inference input.
module HyNomasConditionedFourProbe
  extend self
  alias Four = HyNomasFourBlockProbe
  alias Serial = HyNomasPretrainedSerialProbe
  CONFIG            = Four::CONFIG
  SCHEMA            = "hy-nomas-conditioned-four-reference-v1"
  TEXT_SHA          = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
  TEXT_MANIFEST_SHA = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
  VALID_TEXT        =              23
  VALID_JOINT       =            5303
  SIDECAR_BYTES     = 274_112_512_i64
  INPUT_BYTES       = (5280_i64 * 2048 + 2_i64 * 5280 * 128) * 4
  MAX_SESSION_BYTES = 7_i64 * 1024 * 1024 * 1024
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    raw_bit_differences: Int32, max_abs: Float64?, finite: Bool,
    native_sha256: String, first_request_sha256_match: Bool?)

  def shapes : Hash(String, Array(Int32))
    result = {"input.img" => [1, 5280, 2048],
              "input.rope_cos" => [5280, 128], "input.rope_sin" => [5280, 128],
              "conditioning.projected_text" => [1, 1280, 2048],
              "conditioning.vec" => [1, 2048]}
    4.times do |index|
      result["block#{index}.img.output"] = [1, 5280, 2048]
      result["block#{index}.txt.output"] = [1, 1280, 2048]
    end
    result
  end

  def validate_document!(doc : JSON::Any, qwen_sha : String) : Nil
    unless doc["schema"].as_s == SCHEMA && doc["schema_version"].as_i == 1 &&
           doc["upstream"]["revision"].as_s == Serial::PINNED_UPSTREAM_REVISION
      raise ArgumentError.new("conditioned four-block schema/upstream mismatch")
    end
    {"batches" => 1, "video_tokens" => 5280, "text_tokens" => 1280,
     "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192,
     "query_chunk" => 256, "attention_segments" => 1, "chunk_size_tokens" => 0}.each do |name, value|
      raise ArgumentError.new("conditioned four-block geometry mismatch: #{name}") unless doc["geometry"][name].as_i == value
    end
    {"valid_text_tokens" => VALID_TEXT, "padded_text_tokens" => 1257,
     "valid_joint_tokens" => VALID_JOINT, "refiner_input_dim" => 3584,
     "refiner_depth" => 2}.each do |name, value|
      raise ArgumentError.new("conditioned padding geometry mismatch: #{name}") unless doc["geometry"][name].as_i == value
    end
    g = doc["geometry"]
    unless g["rope_grid"].as_a.map(&.as_i) == [6, 22, 40] &&
           g["rope_axis_dims"].as_a.map(&.as_i) == [16, 56, 56] && g["rope_theta"].as_f == 256.0
      raise ArgumentError.new("conditioned four-block RoPE mismatch")
    end
    unless doc["tolerance"]["atol"].as_f == Serial::ATOL && doc["tolerance"]["rtol"].as_f == Serial::RTOL &&
           doc["precision"]["compute"].as_s == "F32" &&
           doc["precision"]["parameter_storage"].as_s == "F32" &&
           doc["precision"]["attention"].as_s == "PyTorch MPS SDPA F32" &&
           doc["precision"]["parameter_values"].as_s == "BF16-rounded values"
      raise ArgumentError.new("conditioned four-block precision/unchanged tolerance mismatch")
    end
    receipt = doc["producer"]
    {"conditioning_invocations" => 1, "refiner_rows" => 1280, "refiner_attention_calls" => 2,
     "attention_adapter_calls" => 4, "attention_query_chunks_per_block" => 21,
     "attention_query_chunk_calls" => 84, "terminal_tensor_count" => 13,
     "padding_poison_control_calls" => 1, "padding_poison_control_query_chunks" => 21}.each do |name, value|
      raise ArgumentError.new("conditioned producer receipt mismatch: #{name}") unless receipt[name].as_i == value
    end
    unless receipt["device"].as_s == "mps" &&
           receipt["activation_readback"].as_s == "one packed final F32 tensor sidecar readback" &&
           receipt["block_invocations"].as_a.map(&.as_s) == ["block0", "block1", "block2", "block3"] &&
           receipt["captures_per_block"].as_a.map(&.as_s) == Four::CAPTURE_NAMES &&
           receipt["padding_poison_control_passed"].as_bool &&
           receipt["block_handoff"]["same_mps_tensor_objects"].as_bool &&
           receipt["terminal_sidecar_bytes"].as_i64 == SIDECAR_BYTES
      raise ArgumentError.new("conditioned producer device/readback/handoff/control receipt mismatch")
    end
    q = doc["qwen_source"]
    unless qwen_sha.matches?(/\A[0-9a-f]{64}\z/) && q["reference_sha256"].as_s == qwen_sha &&
           q["sidecar_sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/) && q["state_index"].as_i == 26 &&
           q["crop"].as_a.map(&.as_i) == [108, 1388] && doc["timestep"].as_f == 500.0 &&
           doc["mask"].as_a.map(&.as_i) == Array(Int32).new(1280) { |i| i < VALID_TEXT ? 1 : 0 }
      raise ArgumentError.new("conditioned four-block Qwen crop/mask/timestep mismatch")
    end
    p = doc["padding_policy"]
    unless p["kind"].as_s == "unpad_zero_tokens" && p["valid_joint_prefix"].as_i == VALID_JOINT &&
           p["valid_text_rows"].as_i == VALID_TEXT && p["padded_text_rows"].as_i == 1257 && p["refiner_rows"].as_i == 1280
      raise ArgumentError.new("conditioned four-block padding policy mismatch")
    end
    unless doc["text_source"]["sha256"].as_s == TEXT_SHA && doc["text_source"]["manifest_sha256"].as_s == TEXT_MANIFEST_SHA &&
           doc["text_source"]["nbytes"].as_i64 == 583_316_840_i64 && doc["text_source"]["tensor_count"].as_i == 42
      raise ArgumentError.new("conditioned four-block refiner source mismatch")
    end
    source = doc["source"]
    unless source["block_order"].as_a.map(&.as_s) == ["block0", "block1", "block2", "block3"] &&
           source["blocks"].as_h.keys.to_set == Set{"block0", "block1", "block2", "block3"}
      raise ArgumentError.new("conditioned four-block source order/inventory mismatch")
    end
    Four::SOURCE_PINS.each_with_index do |pin, index|
      block = source["blocks"]["block#{index}"]
      unless block["block_index"].as_i == index && block["file_sha256"].as_s == pin[:file_sha256] &&
             block["manifest_sha256"].as_s == pin[:manifest_sha256] &&
             block["tensor_prefix"].as_s == "double_blocks.#{index}." && block["tensor_count"].as_i == 32
        raise ArgumentError.new("conditioned four-block source pin mismatch: #{index}")
      end
    end
    expected = shapes
    tensors = doc["tensors"].as_h
    raise ArgumentError.new("conditioned four-block requires thirteen selected tensors") unless tensors.keys.to_set == expected.keys.to_set
    cursor = 0_i64
    expected.keys.sort.each do |name|
      meta = tensors[name]
      bytes = expected[name].reduce(1_i64) { |a, b| a * b } * 4
      unless meta["dtype"].as_s == "F32" && meta["shape"].as_a.map(&.as_i) == expected[name] &&
             meta["offset_bytes"].as_i64 == cursor && meta["nbytes"].as_i64 == bytes
        raise ArgumentError.new("conditioned four-block tensor layout mismatch: #{name}")
      end
      cursor += bytes
    end
    unless cursor == SIDECAR_BYTES && doc["tensor_file"]["nbytes"].as_i64 == cursor &&
           doc["tensor_file"]["encoding"].as_s == "f32-le" && doc["tensor_file"]["sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("conditioned four-block sidecar extent/encoding mismatch")
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
      HyNomasConditionedFourProbe.validate_document!(@document, qwen_sha)
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

  # Header/digest-only admission, before the live Qwen producer allocates Metal.
  def admit(fixture : Fixture, sources : Array(Serial::BF16Source), budget : Int64, mode : String) : Int64
    raise ArgumentError.new("unknown conditioned four-block control") unless {"positive", "padding-as-valid", "block3-as-block2"}.includes?(mode)
    Four.validate_sources!(sources)
    files = [] of ML::Video::CheckpointFile
    begin
      weights_bytes = 0_i64
      sources.each_with_index do |source, index|
        raise ArgumentError.new("conditioned cache is not the qualified block #{index}") unless source[:sha256] == Four::CACHE_SHAS[index]
        file = ML::Video::BF16WeightCache.authenticate(source[:path], source[:sha256])
        files << file
        plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, budget,
          resident_dtype: ML::Video::WeightDType::BF16, block_index: index.to_i32, require_all: true)
        Four.validate_bf16_plan!(plan, index.to_i32)
        weights_bytes += plan.peak_bytes
      end
      text = ML::Video::CheckpointFile.new(fixture.text_weights)
      files << text
      text_plan = ML::Video::WeightPlan.new([text], ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
        budget, resident_dtype: ML::Video::WeightDType::BF16, require_all: true, allow_f32_to_bf16: true)
      attention_tokens = mode == "padding-as-valid" ? 6560 : VALID_JOINT
      sizes = Four.named_bytes(weights_bytes)
      old_attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, 6560, 16, 128, 256, 16).resident_bytes
      attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1, attention_tokens, 16, 128, 256, 16).resident_bytes
      total = sizes[:total] - Four::INPUT_BYTES + INPUT_BYTES - old_attention + attention + text_plan.peak_bytes +
              ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(fixture.mask, retain_padding: true).total
      unless 0 < total <= budget && total <= MAX_SESSION_BYTES
        raise ArgumentError.new("conditioned four-block exceeds aggregate named-byte budget or 7 GiB session limit")
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
      @named_bytes = HyNomasConditionedFourProbe.admit(@fixture, sources, budget, @mode)
      begin
        @@live << self
        sources.each_with_index do |item, index|
          file = ML::Video::BF16WeightCache.authenticate(item[:path], item[:sha256])
          @files << file
          plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, budget,
            resident_dtype: ML::Video::WeightDType::BF16, block_index: index.to_i32, require_all: true)
          Four.validate_bf16_plan!(plan, index.to_i32)
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
        used[3] = @weights[2] if @mode == "block3-as-block2"
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
        expected = HyNomasConditionedFourProbe.shapes.keys.reject(&.starts_with?("input.")).sort
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
            when "block3-as-block2" then 3
            else                         return false
            end
    (0...fault).all? do |index|
      rows.select { |row| row[:name].starts_with?("block#{index}.") }.all? { |row| row[:failures] == 0 }
    end && rows.select { |row| row[:name].starts_with?("block#{fault}.") }.all? { |row| row[:failures] > 0 }
  end
end
