require "./hy_nomas_pretrained_serial_probe"

# Correctness-only full-J replay. The upstream file stores six inputs and just
# eight checkpoints. Neural execution uses the existing native block path.
module HyNomasFullJProbe
  extend self
  alias Serial = HyNomasPretrainedSerialProbe
  SCHEMA = "hy-nomas-full-j-reference-v1"
  CONFIG = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
    text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192,
    attention_precision: ML::Video::HyNomasAttentionPrecision::F32Control)
  QUERY_CHUNK  = 256
  HEAD_GROUP   =  16
  INPUT_SHAPES = {
    "input.img" => [1, 5280, 2048], "input.txt" => [1, 1280, 2048],
    "input.vec" => [1, 2048], "input.vec_txt" => [1, 2048],
    "input.rope_cos" => [5280, 128], "input.rope_sin" => [5280, 128],
  }
  CAPTURE_NAMES = ["img.attn.q_rope", "img.attn.k_rope", "img.output", "txt.output"]
  ROPE_NAMES    = ["img.attn.q_rope", "img.attn.k_rope"]
  CACHE_SHAS    = [
    "2faa25f5cd7c8392f7f0004146acb10da9ddac062297bc05f701f203107b5c86",
    "2d8049ed5e1f950716bcdc9b3cd7ee6349015c4031085c86d720d47e284bca73",
  ]
  SIDECAR_BYTES          = 339_656_704_i64
  INPUT_BYTES            =  59_162_624_i64
  ROPE_SNAPSHOT_BYTES    = 173_015_040_i64
  SELECTED_CAPTURE_BYTES = 280_494_080_i64
  MAX_NAMED_BYTES        = 5_i64 * 1024 * 1024 * 1024

  def shapes : Hash(String, Array(Int32))
    result = INPUT_SHAPES.dup
    2.times do |index|
      CAPTURE_NAMES.each do |name|
        shape = case name
                when "img.attn.q_rope", "img.attn.k_rope" then [1, 5280, 16, 128]
                when "img.output"                         then [1, 5280, 2048]
                else                                           [1, 1280, 2048]
                end
        result["block#{index}.#{name}"] = shape
      end
    end
    result
  end

  # CPU-only preflight, also exercised without constructing a Metal device.
  def validate_document!(document : JSON::Any) : Nil
    unless document["schema"].as_s == SCHEMA && document["schema_version"].as_i == 1 &&
           document["upstream"]["revision"].as_s == Serial::PINNED_UPSTREAM_REVISION
      raise ArgumentError.new("full-J reference schema/upstream revision mismatch")
    end
    geometry = document["geometry"]
    {"batches" => 1, "video_tokens" => 5280, "text_tokens" => 1280,
     "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192,
     "query_chunk" => 256, "attention_segments" => 1, "chunk_size_tokens" => 0}.each do |name, value|
      raise ArgumentError.new("full-J geometry mismatch: #{name}") unless geometry[name].as_i == value
    end
    unless geometry["rope_grid"].as_a.map(&.as_i) == [6, 22, 40] &&
           geometry["rope_axis_dims"].as_a.map(&.as_i) == [16, 56, 56] &&
           geometry["rope_theta"].as_f == 256.0
      raise ArgumentError.new("full-J ND RoPE geometry mismatch")
    end
    unless document["tolerance"]["atol"].as_f == Serial::ATOL &&
           document["tolerance"]["rtol"].as_f == Serial::RTOL &&
           document["precision"]["compute"].as_s == "F32" &&
           document["precision"]["parameter_storage"].as_s == "F32" &&
           document["precision"]["attention"].as_s == "PyTorch MPS SDPA F32" &&
           document["precision"]["parameter_values"].as_s == "BF16-rounded values"
      raise ArgumentError.new("full-J precision or unchanged tolerance mismatch")
    end
    producer = document["producer"]
    unless producer["device"].as_s == "mps" &&
           producer["activation_readback"].as_s == "one packed final F32 tensor sidecar readback" &&
           producer["attention_adapter_calls"].as_i == 2 &&
           producer["attention_query_chunks_per_block"].as_i == 26 &&
           producer["terminal_tensor_count"].as_i == 14 &&
           producer["captures_per_block"].as_a.map(&.as_s) == CAPTURE_NAMES
      raise ArgumentError.new("full-J producer execution receipt mismatch")
    end
    {"block0" => Serial::BLOCK0_PIN, "block1" => Serial::BLOCK1_PIN}.each do |name, pin|
      block = document["source"]["blocks"][name]
      unless block["block_index"].as_i == pin.index &&
             block["file_sha256"].as_s == pin.file_sha256 &&
             block["manifest_sha256"].as_s == pin.manifest_sha256 &&
             block["tensor_count"].as_i == 32
        raise ArgumentError.new("full-J upstream checkpoint provenance mismatch: #{name}")
      end
    end
    tensors = document["tensors"].as_h
    expected = shapes
    raise ArgumentError.new("full-J reference requires exactly fourteen selected tensors") unless tensors.keys.to_set == expected.keys.to_set
    cursor = 0_i64
    expected.keys.sort.each do |name|
      metadata = tensors[name]
      size = expected[name].reduce(1_i64) { |a, b| a * b } * 4
      unless metadata["dtype"].as_s == "F32" &&
             metadata["shape"].as_a.map(&.as_i) == expected[name] &&
             metadata["offset_bytes"].as_i64 == cursor && metadata["nbytes"].as_i64 == size
        raise ArgumentError.new("full-J reference layout mismatch: #{name}")
      end
      cursor += size
    end
    sidecar = document["tensor_file"]
    unless cursor == SIDECAR_BYTES && sidecar["nbytes"].as_i64 == cursor &&
           sidecar["encoding"].as_s == "f32-le" &&
           sidecar["sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("full-J sidecar extent/encoding/digest mismatch")
    end
  end

  def named_bytes(weight_bytes : Int64) : NamedTuple(scratch: Int64, workspace: Int64, total: Int64)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG).sum { |_, n| n.to_i64 * 4 }
    scratch += CONFIG.joint_rows.to_i64 * CONFIG.dim * 4 # Distinct successor output banks.
    compact = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
      1, 6560, 16, 128, QUERY_CHUNK, HEAD_GROUP)
    workspace = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(CONFIG).to_i64 * 4 +
                compact.resident_bytes + 2_i64 * CONFIG.joint_rows * (CONFIG.dim + CONFIG.heads) * 4
    {scratch: scratch, workspace: workspace,
     total: weight_bytes + INPUT_BYTES + scratch + workspace + ROPE_SNAPSHOT_BYTES}
  end

  def admit_budget!(required : Int64, budget : Int64) : Nil
    unless 0 < required <= budget <= MAX_NAMED_BYTES
      raise ArgumentError.new("full-J aggregate named-byte budget is insufficient or exceeds the 5 GiB limit")
    end
  end

  # Publish a completed report without replacing a concurrent writer's file.
  # Crystal's File mode strings do not support the POSIX/Python "x" suffix.
  def write_report_new!(report : String, json : String) : Nil
    temporary = File.tempfile(".hy-nomas-full-j", ".json", dir: File.dirname(File.expand_path(report)))
    begin
      temporary << json << '\n'
      temporary.close
      File.link(temporary.path, report)
    ensure
      temporary.close
      temporary.delete
    end
  end

  class Fixture
    getter document : JSON::Any
    getter sha256 : String
    @sidecar : String
    @info : File::Info

    def initialize(path : String, expected_sha : String)
      unless File.file?(path) && !File.symlink?(path) &&
             0 < File.size(path) <= 2 * 1024 * 1024 && expected_sha.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("full-J reference must be a bounded regular JSON file with SHA256 pin")
      end
      raw = File.read(path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("full-J reference digest mismatch") unless @sha256 == expected_sha
      @document = JSON.parse(raw)
      HyNomasFullJProbe.validate_document!(@document)
      sidecar = @document["tensor_file"]
      name = sidecar["name"].as_s
      unless name.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.f32le\z/)
        raise ArgumentError.new("full-J sidecar requires a sibling basename")
      end
      @sidecar = File.join(File.dirname(File.realpath(path)), name)
      unless File.file?(@sidecar) && !File.symlink?(@sidecar) && File.size(@sidecar) == SIDECAR_BYTES
        raise ArgumentError.new("full-J sidecar file extent/type mismatch")
      end
      @info = File.info(@sidecar)
      raise ArgumentError.new("full-J sidecar digest mismatch") unless Serial.sha256_file(@sidecar) == sidecar["sha256"].as_s
      unchanged!
    end

    def unchanged! : Nil
      info = File.info(@sidecar, follow_symlinks: false)
      unless info.file? && @info.same_file?(info) && info.size == @info.size &&
             info.modification_time == @info.modification_time
        raise ArgumentError.new("full-J sidecar changed during use")
      end
    end

    # Setup upload only. Keep at most one input tensor on the host at a time.
    def values(name : String) : Array(Float32)
      metadata = @document["tensors"][name]
      size = metadata["nbytes"].as_i64.to_i32
      raw = Bytes.new(size)
      read_tensor(name) { |io| io.read_fully(raw) }
      Array(Float32).new(size // 4) { |i| IO::ByteFormat::LittleEndian.decode(Float32, raw[i * 4, 4]) }
    end

    def read_tensor(name : String, & : File ->) : Nil
      unchanged!
      File.open(@sidecar, "r") do |io|
        io.seek(@document["tensors"][name]["offset_bytes"].as_i64)
        yield io
      end
      unchanged!
    end
  end

  class RunOwner
    @@live = [] of RunOwner
    @inputs : ML::Video::Arena? = nil
    @snapshots : ML::Video::Arena? = nil
    @weights = [] of ML::Video::Arena
    @files = [] of ML::Video::CheckpointFile
    @plans = [] of ML::Video::WeightPlan
    @prepared = [] of ML::Video::PreparedWeights
    @serial : ML::Video::HyNomasMPSSerialWorkspace? = nil
    @captures = {} of String => ML::MetalBuffer
    @scratch_to_poison = [] of ML::MetalBuffer
    @poison : ML::Metal::ComputePipeline? = nil
    @identity : ML::Metal::ComputePipeline? = nil
    @chain : ML::Metal::MPSCommandChain? = nil
    @lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @released = false
    @first_hashes = {} of String => String
    @diagnostic = [] of Float32
    @expected_slab = Bytes.new(64 * 1024)
    getter estimate : NamedTuple(scratch: Int64, workspace: Int64, total: Int64)

    def initialize(@reference : Fixture, sources : Array(Serial::BF16Source), budget : Int64, @mode : String)
      @estimate = HyNomasFullJProbe.named_bytes(604_227_584_i64)
      HyNomasFullJProbe.admit_budget!(@estimate[:total], budget)
      raise ArgumentError.new("unknown full-J probe mode") unless {"positive", "identity-rope", "successor-block0"}.includes?(@mode)
      Serial.validate_bf16_sources!(sources)
      begin
        sources.each_with_index do |source, index|
          raise ArgumentError.new("full-J cache is not the qualified block #{index} derivative") unless source[:sha256] == CACHE_SHAS[index]
          file = ML::Video::BF16WeightCache.authenticate(source[:path], source[:sha256])
          @files << file
          plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, budget,
            resident_dtype: ML::Video::WeightDType::BF16, block_index: index.to_i32,
            require_all: true, allow_f32_to_bf16: false)
          Serial.validate_bf16_plan!(plan, index.to_i32)
          @plans << plan
        end
        raise "full-J weight-byte estimate mismatch" unless @plans.sum(&.destination_bytes) == 604_227_584_i64
        @@live << self # Root the complete owner before its first Metal allocation.
        @plans.each do |plan|
          arena = ML::Video::Arena.new
          @weights << arena
          @prepared << ML::Video::PreparedWeights.new(plan, arena)
        end
        input = ML::Video::Arena.new
        @inputs = input
        INPUT_SHAPES.each_key do |name|
          values = @reference.values(name)
          raise ArgumentError.new("non-finite full-J input: #{name}") unless values.all?(&.finite?)
          input.upload(name, values)
        end
        raise "full-J input-byte estimate mismatch" unless input.resident_bytes == INPUT_BYTES
        weights = @mode == "successor-block0" ? [@weights[0], @weights[0]] : @weights
        serial = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG, weights,
          input["input.img"], input["input.txt"], input["input.vec"], input["input.vec_txt"],
          input["input.rope_cos"], input["input.rope_sin"],
          attention_query_chunk: QUERY_CHUNK, attention_heads_per_group: HEAD_GROUP)
        @serial = serial
        unless serial.scratch_bytes == @estimate[:scratch] && serial.mps_workspace_bytes == @estimate[:workspace]
          raise "full-J shared workspace differs from pre-allocation byte estimate"
        end
        snapshots = ML::Video::Arena.new
        @snapshots = snapshots
        serial.blocks.each_with_index do |block, index|
          CAPTURE_NAMES.each do |name|
            source = block.checkpoints[name]
            key = "block#{index}.#{name}"
            raise "full-J checkpoint extent changed: #{key}" unless source.size == @reference.document["tensors"][key]["nbytes"].as_i64
            @captures[key] = ROPE_NAMES.includes?(name) ? snapshots.alloc(key, (source.size // 4).to_i32) : source
          end
          scratch, _ = block.mps_workspace_buffers
          plan = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG)
          raise "full-J scratch inventory changed" unless scratch.size == plan.size
          scratch.each_with_index do |buffer, slot|
            next if plan[slot][0] == "hy_nomas.zero_head_bias"
            @scratch_to_poison << buffer unless @scratch_to_poison.any?(&.same?(buffer))
          end
        end
        raise "full-J RoPE snapshot byte estimate mismatch" unless snapshots.resident_bytes == ROPE_SNAPSHOT_BYTES
        @diagnostic = Array(Float32).new(5280 * 2048, 0_f32)
        @poison = ML::Metal::ComputePipeline.new("hy_nomas_full_j_poison", <<-METAL)
          #include <metal_stdlib>
          using namespace metal;
          kernel void hy_nomas_full_j_poison(device float *p [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i < n) p[i] = as_type<float>(0x7fc00000u);
          }
          METAL
        @identity = ML::Metal::ComputePipeline.new("hy_nomas_full_j_identity", <<-METAL)
          #include <metal_stdlib>
          using namespace metal;
          kernel void hy_nomas_full_j_identity(device float *c [[buffer(0)]], device float *s [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < n) { c[i] = 1.0f; s[i] = 0.0f; }
          }
          METAL
      rescue ex
        cleanup!
        raise ex
      end
    end

    def request(index : Int32)
      serial = @serial.not_nil!
      chain = ML::Metal::MPSCommandChain.new
      @chain = chain
      @lease.begin_encode!(chain)
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      @prepared.each { |p| p.encode(chain.encoder) } if index == 0
      @scratch_to_poison.each do |buffer|
        chain.encoder.set_pipeline(@poison.not_nil!).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
          .set_value((buffer.size // 4).to_u32, 1).dispatch_1d((buffer.size // 4).to_i32)
      end
      if @mode == "identity-rope"
        inputs = @inputs.not_nil!
        chain.encoder.set_pipeline(@identity.not_nil!)
          .set_buffer(inputs["input.rope_cos"], 0, ML::Metal::BufferAccess::Write)
          .set_buffer(inputs["input.rope_sin"], 1, ML::Metal::BufferAccess::Write)
          .set_value((5280 * 128).to_u32, 2).dispatch_1d(5280 * 128)
      end
      copies = 0_i32
      callback = ->(block_index : Int32, block : ML::Video::HyNomasDualStreamBlock) do
        ROPE_NAMES.each do |name|
          source = block.checkpoints[name]
          target = @captures["block#{block_index}.#{name}"]
          raise "RoPE diagnostic snapshot aliases shared scratch" if source.same?(target)
          ML::Video::MetalOps.copy(chain.encoder, source, target, (source.size // 4).to_i32)
          copies += 1
        end
        nil
      end
      serial.encode(chain, after_block: callback)
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      pre_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      raise "full-J graph allocated buffers, read activations, or waited before terminal" unless allocations == 0 && pre_reads == 0 && chain.wait_count == 0 && copies == 4
      chain.commit_and_wait
      raise "full-J graph requires one successful terminal wait" unless chain.completed? && chain.wait_count == 1
      @lease.ensure_terminal!
      waits = chain.wait_count
      chain.release
      @chain = nil
      rows = @captures.keys.sort.map { |key| compare(key, @captures[key], index) }
      diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      raise "full-J selected readback count mismatch" unless diagnostic_reads == 8
      {request: index, tensors: rows, allocations_during_encode: allocations,
       readbacks_before_terminal_wait: pre_reads, terminal_wait_count: waits,
       diagnostic_readbacks_after_terminal_wait: diagnostic_reads, snapshot_copy_dispatches: copies,
       poison_dispatches: @scratch_to_poison.size, owner_rooted_through_terminal: @@live.any?(&.same?(self))}
    rescue ex
      drain!
      raise ex
    end

    private def compare(name : String, buffer : ML::MetalBuffer, request : Int32)
      count = (buffer.size // 4).to_i32
      actual = Slice.new(@diagnostic.to_unsafe, count)
      buffer.read_to_slice(actual)
      hash = Digest::SHA256.hexdigest(Bytes.new(actual.to_unsafe.as(UInt8*), count * 4))
      repeat_match = request == 0 ? nil : @first_hashes[name] == hash
      @first_hashes[name] = hash if request == 0
      failures = 0_i32
      raw_differences = 0_i32
      max_abs = 0_f64
      cursor = 0
      @reference.read_tensor(name) do |io|
        while cursor < count
          take = Math.min(@expected_slab.size // 4, count - cursor)
          slab = @expected_slab[0, take * 4]
          io.read_fully(slab)
          take.times do |offset|
            expected = IO::ByteFormat::LittleEndian.decode(Float32, slab[offset * 4, 4])
            value = actual[cursor + offset]
            raw_differences += 1 if value.unsafe_as(UInt32) != expected.unsafe_as(UInt32)
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
      # A failed finite gate must still produce a JSON-serializable report.
      {name: name, elements: count, failures: failures, raw_bit_differences: raw_differences,
       max_abs: max_abs.finite? ? max_abs : nil, finite: max_abs.finite?,
       native_sha256: hash, first_request_sha256_match: repeat_match}
    end

    def drain! : Nil
      if chain = @chain
        Qwen25PrefixProbe.cleanup_after_drain(chain) do
          @lease.ensure_terminal!
          chain.release
          @chain = nil
        end
      end
    end

    def cleanup! : Nil
      return if @released
      raise "retain full-J owner until pending GPU work drains" if @chain
      @serial.try(&.release)
      @serial = nil
      @captures.clear
      @scratch_to_poison.clear
      @snapshots.try(&.release)
      @inputs.try(&.release)
      @prepared.each(&.release)
      @files.each(&.close)
      @weights.each(&.release)
      @released = true
      @@live.reject!(&.same?(self))
    end
  end

  def run(reference : String, sha : String, sources : Array(Serial::BF16Source),
          report : String, budget : Int64, mode : String) : Bool
    raise ArgumentError.new("refusing to replace full-J report") if File.exists?(report) || File.symlink?(report)
    raise ArgumentError.new("full-J report parent must exist") unless Dir.exists?(File.dirname(File.expand_path(report)))
    fixture = Fixture.new(reference, sha)
    owner = nil.as(RunOwner?)
    begin
      owner = RunOwner.new(fixture, sources, budget, mode)
      requests = [owner.request(0)]
      requests << owner.request(1) if mode == "positive"
      rows = requests.flat_map { |r| r[:tensors] }
      upstream_passed = rows.all? { |r| r[:finite] && r[:failures] == 0 }
      repeat_passed = mode == "positive" ? requests.size == 2 &&
                                           requests[1][:tensors].all? { |r| r[:first_request_sha256_match] == true } : nil
      detected = if mode == "successor-block0"
                   rows.select(&.[:name].starts_with?("block0.")).all? { |r| r[:failures] == 0 } &&
                     rows.any? { |r| r[:name].starts_with?("block1.") && r[:name].ends_with?(".output") && r[:failures] > 0 }
                 elsif mode == "identity-rope"
                   rows.select { |r| r[:name].starts_with?("block0.img.attn.") }.all? { |r| r[:failures] > 0 } &&
                     rows.any? { |r| r[:name].ends_with?(".output") && r[:failures] > 0 }
                 else
                   false
                 end
      detected = detected && rows.all?(&.[:finite])
      passed = mode == "positive" ? upstream_passed && repeat_passed == true : detected
      result = {schema: "cogni_ml.hy_nomas_full_j_probe", schema_version: 1,
                mode: mode, passed: passed, upstream_passed: upstream_passed,
                repeat_sha256_passed: repeat_passed, negative_control_detected: detected,
                reference_sha256: fixture.sha256, caches: sources,
                tolerance: {atol: Serial::ATOL, rtol: Serial::RTOL},
                named_bytes: owner.estimate, input_bytes: INPUT_BYTES,
                rope_snapshot_bytes: ROPE_SNAPSHOT_BYTES, selected_capture_bytes: SELECTED_CAPTURE_BYTES,
                capture_tensor_count: 8, source_staging_bytes: 0, source_conversion_dispatches: 0,
                parameter_storage: "BF16", compute: "F32", attention_query_chunk: QUERY_CHUNK,
                attention_heads_per_group: HEAD_GROUP, requests: requests,
                scope: "same-input two-block full-J replay; synthetic hidden inputs; upstream ND RoPE table upload and native RoPE application; no generation or performance claim"}
      owner.cleanup!
      write_report_new!(report, result.to_pretty_json)
      puts({report: report, passed: passed, upstream_passed: upstream_passed,
            negative_control_detected: detected}.to_json)
      passed
    ensure
      if current = owner
        current.drain!
        current.cleanup!
      end
    end
  end
end
